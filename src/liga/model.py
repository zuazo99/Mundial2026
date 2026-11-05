"""Modelo de XG para La Liga (clubes).

Reutiliza el pipeline de xg_preds (ventanas móviles, PCA de estilo, XGBoost
Tweedie, rho de Dixon-Coles) con estas diferencias:
  * decaimiento temporal anclado a la última fecha de entrenamiento;
  * la primera temporada no entra en el ajuste (ELO en frío), pero se usa para
    las ventanas de la segunda;
  * is_home es real; `confed` y `tournament_num` son constantes (no informan);
  * un único modelo "liga" (no hay variantes de ELO como en el Mundial).

Limitación: el "estado" de cada equipo (ventanas, PCA) se congela tras su último
partido jugado y se usa igual para todos los pendientes. La fase de simulación
podrá actualizarlo jornada a jornada.

Uso:  python src/liga/model.py      # tras python src/liga/ingest.py
"""
import glob
import json
import os
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FEATURES, TARGET, path  # noqa: E402
from xg_preds import (_CONFED_NUMERIC, _DEFAULT_PARAMS, _add_rolling_and_style,  # noqa: E402
                      _apply_pca_merge_decay, engineer_features, estimate_rho)

# Elecciones de modelado (ajustables):
HALF_LIFE_DAYS = 730          # vida media del peso temporal (2 años)
MODEL_NAME = "liga"
_STATE_COLS = ["gf_prom_5", "gc_prom_5", "elo_prom_5", "gf_prom_15", "gc_prom_15", "PCA_1", "PCA_2"]
_NEXT = "__NEXT__"
_CONFED_CONST = _CONFED_NUMERIC["OTHER"]   # 0 para todos los clubes
_TOURNAMENT_CONST = 1                      # igual que el entrenamiento (torneo sin mapear)


@dataclass
class LigaModel:
    model: xgb.XGBRegressor
    scaler: object
    pca: object
    feat_means: pd.Series
    rho: float
    decay_ref: pd.Timestamp
    half_life_days: int
    train_seasons: list = field(default_factory=list)
    n_train_rows: int = 0


def _sorted(long_df: pd.DataFrame) -> pd.DataFrame:
    return long_df.sort_values(["team", "date"], kind="stable").reset_index(drop=True)


def _check_unique_dates(df: pd.DataFrame) -> None:
    """El cruce con el rival se hace por (fecha, equipo): un equipo no puede jugar dos veces el mismo día."""
    dup = df[df.duplicated(["date", "team"], keep=False)]
    if len(dup):
        raise ValueError(f"{len(dup)} filas de equipos con dos partidos en la misma fecha "
                         f"(p. ej. {dup.iloc[0]['team']} el {dup.iloc[0]['date'].date()}); revisa los CSV.")


def _force_club_constants(df: pd.DataFrame) -> pd.DataFrame:
    """confed/tournament_num se derivan del nombre del equipo o del torneo (selecciones);
    para clubes deben ser constantes también al entrenar (p. ej. 'Andorra' está en la lista UEFA)."""
    df["confed"] = _CONFED_CONST
    df["rival_confed"] = _CONFED_CONST
    df["tournament_num"] = _TOURNAMENT_CONST
    return df


def _fit_xgb(df_train: pd.DataFrame, feat_means: pd.Series) -> xgb.XGBRegressor:
    x = df_train[FEATURES].fillna(feat_means)
    m = xgb.XGBRegressor(**_DEFAULT_PARAMS)
    m.fit(x, df_train[TARGET], sample_weight=df_train["date_weight"])
    return m


def train_liga(long_df: pd.DataFrame, half_life_days: int = HALF_LIFE_DAYS) -> LigaModel:
    """Ajusta el modelo con todas las temporadas menos la primera."""
    df = _sorted(long_df)
    _check_unique_dates(df)
    seasons = sorted(df["season"].unique())
    decay_ref = df["date"].max()
    eng, scaler, pca, _ = engineer_features(df, fit_transformers=True,
                                            decay_ref=decay_ref, half_life_days=half_life_days)
    train = _force_club_constants(eng[eng["season"] != seasons[0]].dropna(subset=[TARGET]).copy())
    feat_means = train[FEATURES].mean()
    train[FEATURES] = train[FEATURES].fillna(feat_means)
    model = _fit_xgb(train, feat_means)
    return LigaModel(model, scaler, pca, feat_means, estimate_rho(model, train), decay_ref,
                     half_life_days, train_seasons=seasons[1:], n_train_rows=len(train))


# ─── estado de cada equipo y predicción ──────────────────────────────────────

def team_state(long_df: pd.DataFrame, lm: LigaModel, elo_now: dict) -> pd.DataFrame:
    """Estado actual de cada equipo con ELO actual: ventanas y PCA tras su último partido.

    Se añade una fila futura por equipo (goles NaN, rival marcador) cuyas ventanas
    (shift(1)) resumen los partidos ya jugados. Se usan el scaler y el PCA ya ajustados.
    """
    df = _sorted(long_df)
    teams = sorted(t for t in elo_now if t in set(df["team"]))
    fut = pd.DataFrame({
        "date": df["date"].max() + pd.Timedelta(days=1), "team": teams, "opponent": _NEXT,
        "goals": np.nan, "goals_conceded": np.nan, "is_home": 0,
        "elo": [elo_now[t] for t in teams], "opponent_elo": np.nan,
        "tournament": df["tournament"].iloc[0], "season": "next",
    })
    both = _sorted(pd.concat([df, fut], ignore_index=True))
    eng, *_ = engineer_features(both, scaler=lm.scaler, pca=lm.pca, fit_transformers=False,
                                decay_ref=lm.decay_ref, half_life_days=lm.half_life_days)
    st = eng[eng["opponent"] == _NEXT].set_index("team")[_STATE_COLS + ["elo"]]
    return st.loc[teams]


def _rows(state: pd.DataFrame, team: pd.Series, rival: pd.Series, is_home: int) -> pd.DataFrame:
    s, r = state.loc[team.values].reset_index(drop=True), state.loc[rival.values].reset_index(drop=True)
    out = pd.DataFrame({
        "elo": s["elo"], "opponent_elo": r["elo"], "is_home": is_home,
        "tournament_num": _TOURNAMENT_CONST, "confed": _CONFED_CONST, "rival_confed": _CONFED_CONST,
    })
    for c in _STATE_COLS:
        out[c] = s[c]
        out[f"rival_{c}"] = r[c]
    return out[FEATURES]


def predict_fixtures(lm: LigaModel, state: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    """XG esperado de local y visitante para cada (home, away)."""
    fx = fixtures.reset_index(drop=True)
    xh = lm.model.predict(_rows(state, fx["home"], fx["away"], 1).fillna(lm.feat_means))
    xa = lm.model.predict(_rows(state, fx["away"], fx["home"], 0).fillna(lm.feat_means))
    return pd.DataFrame({"home": fx["home"], "away": fx["away"],
                         "xg_home": np.round(np.maximum(xh, 1e-3), 3),
                         "xg_away": np.round(np.maximum(xa, 1e-3), 3)})


# ─── validación temporal ─────────────────────────────────────────────────────

def _scores(y, pred) -> dict:
    pred = np.maximum(np.asarray(pred, dtype=float), 1e-6)
    return {"mae": round(float(mean_absolute_error(y, pred)), 4),
            "poisson_deviance": round(float(mean_poisson_deviance(y, pred)), 4)}


def holdout_validation(long_df: pd.DataFrame, test_season: str = None,
                       half_life_days: int = HALF_LIFE_DAYS) -> dict:
    """Entrena con las temporadas anteriores (sin la primera) y evalúa en una posterior.

    Scaler, PCA y medias de relleno se ajustan solo con el train. Por defecto la
    temporada de test es la última completa (≥ 90 % de los partidos de la mayor).
    """
    df = _sorted(long_df)
    _check_unique_dates(df)
    seasons = sorted(df["season"].unique())
    if test_season is None:
        sizes = df.groupby("season").size()
        test_season = [s for s in seasons if sizes[s] >= 0.9 * sizes.max()][-1]
    prev = [s for s in seasons if s < test_season]
    if len(prev) < 2:
        raise ValueError("Se necesitan al menos 3 temporadas (1 de arranque en frío, 1 de train y 1 de test).")

    feat = _add_rolling_and_style(df.copy())        # ventanas causales (shift(1))
    cutoff = df.loc[df["season"].isin(prev), "date"].max()
    f_tr, f_te = feat[feat["season"].isin(prev)].copy(), feat[feat["season"] == test_season].copy()
    tr, scaler, pca = _apply_pca_merge_decay(f_tr, fit_transformers=True,
                                             decay_ref=cutoff, half_life_days=half_life_days)
    te, _, _ = _apply_pca_merge_decay(f_te, scaler=scaler, pca=pca, fit_transformers=False,
                                      decay_ref=cutoff, half_life_days=half_life_days)
    tr = _force_club_constants(tr[tr["season"] != prev[0]].dropna(subset=[TARGET]).copy())
    te = _force_club_constants(te.dropna(subset=[TARGET]).copy())
    means = tr[FEATURES].mean()
    m = _fit_xgb(tr, means)

    y = te[TARGET].to_numpy()
    ha = tr.groupby("is_home")[TARGET].mean()
    return {
        "test_season": test_season, "train_seasons": prev[1:], "n_test_rows": int(len(te)),
        "half_life_days": half_life_days,
        "model": _scores(y, m.predict(te[FEATURES].fillna(means))),
        "baseline_constant": _scores(y, np.full(len(te), tr[TARGET].mean())),
        "baseline_home_away": _scores(y, te["is_home"].map(ha).to_numpy()),
    }


# ─── CLI ─────────────────────────────────────────────────────────────────────

def run(data_dir: str, model_dir: str, results_dir: str) -> None:
    long_path = os.path.join(data_dir, "df_liga.csv")
    pend_files = sorted(glob.glob(os.path.join(data_dir, "pending_*.csv")))
    elo_path = os.path.join(data_dir, "elo_actual.csv")
    if not (os.path.exists(long_path) and pend_files and os.path.exists(elo_path)):
        raise SystemExit("Faltan datos ingeridos en {}. Ejecuta antes: python src/liga/ingest.py".format(data_dir))

    long = pd.read_csv(long_path, parse_dates=["date"], dtype={"season": str})
    pending_path = pend_files[-1]
    code = os.path.splitext(os.path.basename(pending_path))[0].split("_", 1)[1]
    pending = pd.read_csv(pending_path)
    elo_now = pd.read_csv(elo_path).set_index("team")["elo"].to_dict()

    lm = train_liga(long)
    os.makedirs(model_dir, exist_ok=True)
    lm.model.save_model(os.path.join(model_dir, f"xgb_{MODEL_NAME}.json"))
    with open(os.path.join(model_dir, f"rho_{MODEL_NAME}.json"), "w", encoding="utf-8") as f:
        json.dump({"variant": MODEL_NAME, "rho": lm.rho}, f)

    out = predict_fixtures(lm, team_state(long, lm, elo_now), pending)
    out.to_csv(os.path.join(data_dir, f"xg_pending_{code}.csv"), index=False)

    val = holdout_validation(long)
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, f"validation_{MODEL_NAME}.json"), "w", encoding="utf-8") as f:
        json.dump(val, f, indent=2, ensure_ascii=False)

    print(f"✅ modelo → {model_dir}/xgb_{MODEL_NAME}.json  (rho={lm.rho:.4f}, filas={lm.n_train_rows})")
    print(f"✅ {len(out)} partidos pendientes → xg_pending_{code}.csv")
    print(f"   hold-out {val['test_season']}: deviance modelo={val['model']['poisson_deviance']} "
          f"vs constante={val['baseline_constant']['poisson_deviance']}")


if __name__ == "__main__":
    run(path("data", "liga"), path("api", "models"), path("results"))
