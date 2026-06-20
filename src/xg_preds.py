import json
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from scipy.optimize import minimize_scalar


# ─── module-level constants ───────────────────────────────────────────────────

FEATURES = [
    'elo', 'opponent_elo', 'is_home', 'tournament_num', 'confed', 'rival_confed',
    'gf_prom_5', 'gc_prom_5', 'elo_prom_5', 'gf_prom_15', 'gc_prom_15', 'PCA_1', 'PCA_2',
    'rival_gf_prom_5', 'rival_gc_prom_5', 'rival_elo_prom_5', 'rival_gf_prom_15',
    'rival_gc_prom_15', 'rival_PCA_1', 'rival_PCA_2',
]
TARGET = "goals"

_FOTO_NUMERIC = ['gf_prom_5', 'gc_prom_5', 'elo_prom_5', 'gf_prom_15', 'gc_prom_15', 'PCA_1', 'PCA_2']
_N_FOTO = 3  # matches to average for the pre-WC snapshot

_COLUMNS_PCA = ["pace_20", "gf_ratio_20", "clean_20", "underdog_20"]

_CONFED_DICT = {
    'CONMEBOL': ['Argentina', 'Bolivia', 'Brazil', 'Chile', 'Colombia', 'Ecuador', 'Paraguay', 'Peru', 'Uruguay', 'Venezuela'],
    'CONCACAF': ['USA', 'United States', 'Mexico', 'Canada', 'Costa Rica', 'Jamaica', 'Panama', 'Honduras', 'El Salvador', 'Haiti', 'Trinidad and Tobago', 'Guatemala', 'Cuba', 'Curaçao', 'Martinique', 'Guadeloupe', 'Suriname'],
    'UEFA': ['Spain', 'France', 'Germany', 'England', 'Italy', 'Netherlands', 'Portugal', 'Belgium', 'Croatia', 'Denmark', 'Switzerland', 'Serbia', 'Poland', 'Sweden', 'Wales', 'Scotland', 'Czech Republic', 'Austria', 'Hungary', 'Ukraine', 'Turkey', 'Russia', 'Norway', 'Republic of Ireland', 'Northern Ireland', 'Slovakia', 'Romania', 'Greece', 'Bosnia and Herzegovina', 'Finland', 'Iceland', 'Albania', 'Slovenia', 'Montenegro', 'North Macedonia', 'Georgia', 'Israel', 'Bulgaria', 'Armenia', 'Luxembourg', 'Cyprus', 'Kosovo', 'Estonia', 'Lithuania', 'Latvia', 'Moldova', 'Belarus', 'Malta', 'Liechtenstein', 'Andorra', 'San Marino', 'Gibraltar', 'Kazakhstan', 'Faroe Islands', 'Azerbaijan'],
    'CAF': ['Senegal', 'Morocco', 'Nigeria', 'Egypt', 'Ivory Coast', 'Cameroon', 'Ghana', 'Algeria', 'Mali', 'Tunisia', 'Burkina Faso', 'South Africa', 'DR Congo', 'Guinea', 'Cape Verde', 'Zambia', 'Gabon', 'Uganda', 'Equatorial Guinea', 'Gambia', 'Angola', 'Mauritania', 'Namibia', 'Benin', 'Mozambique', 'Togo', 'Tanzania', 'Zimbabwe', 'Malawi', 'Kenya', 'Congo', 'Rwanda', 'Madagascar', 'Central African Republic', 'Sudan', 'Sierra Leone'],
    'AFC': ['Japan', 'Iran', 'South Korea', 'Australia', 'Saudi Arabia', 'Qatar', 'Iraq', 'United Arab Emirates', 'Oman', 'Uzbekistan', 'China', 'China PR', 'Jordan', 'Bahrain', 'Syria', 'Vietnam', 'Palestine', 'Kyrgyzstan', 'India', 'Lebanon', 'Tajikistan', 'Thailand', 'North Korea', 'Philippines', 'Malaysia', 'Kuwait', 'Turkmenistan', 'Hong Kong', 'Indonesia', 'Yemen', 'Afghanistan', 'Singapore', 'Myanmar', 'Maldives', 'Nepal', 'Cambodia'],
    'OFC': ['New Zealand', 'Solomon Islands', 'Fiji', 'New Caledonia', 'Tahiti', 'Vanuatu', 'Papua New Guinea', 'Samoa', 'Tonga', 'Cook Islands'],
}
_CONFED_NUMERIC = {"UEFA": 1, "CONMEBOL": 2, "CONCACAF": 3, "CAF": 4, "AFC": 5, "OFC": 6, "OTHER": 0}

_DEFAULT_PARAMS = dict(
    objective='reg:tweedie',
    tweedie_variance_power=1.3,
    n_estimators=500,
    learning_rate=0.01,
    max_depth=4,
    subsample=0.7,
    colsample_bytree=0.8,
    random_state=42,
)


# ─── helpers ─────────────────────────────────────────────────────────────────

def calculate_window(series: pd.Series, window):
    return series.rolling(window=window, min_periods=1).mean().shift(1)


def _add_rolling_and_style(df):
    """Rolling windows + style columns + confed + tournament encoding.
    Requires df sorted by [team, date]. Mutates and returns df."""
    df["gf_prom_5"]  = df.groupby("team")["goals"].transform(lambda x: calculate_window(x, 5))
    df["gc_prom_5"]  = df.groupby("team")["goals_conceded"].transform(lambda x: calculate_window(x, 5))
    df["elo_prom_5"] = df.groupby("team")["opponent_elo"].transform(lambda x: calculate_window(x, 5))
    df["gf_prom_15"] = df.groupby("team")["goals"].transform(lambda x: calculate_window(x, 15))
    df["gc_prom_15"] = df.groupby("team")["goals_conceded"].transform(lambda x: calculate_window(x, 15))

    df["total_goals"] = df["goals"] + df["goals_conceded"]
    df["pace_20"]     = df.groupby("team")["total_goals"].transform(lambda x: calculate_window(x, 20))
    df["gf_ratio"]    = df["goals"] / (df["total_goals"] + 1e-9)
    df["gf_ratio"]    = np.where(df["total_goals"] == 0, 0.5, df["gf_ratio"])
    df["gf_ratio_20"] = df.groupby("team")["gf_ratio"].transform(lambda x: calculate_window(x, 20))
    df["clean_ratio"] = (df["goals_conceded"] == 0).astype(int)
    df["clean_20"]    = df.groupby("team")["clean_ratio"].transform(lambda x: calculate_window(x, 20))
    df["dif_goals_elo"]  = np.where(df["opponent_elo"] > df["elo"], df["goals"] - df["goals_conceded"], np.nan)
    df["underdog_20"]    = df.groupby("team")["dif_goals_elo"].transform(lambda x: calculate_window(x, 20))
    df["underdog_20"]    = df["underdog_20"].fillna(0)  # pandas-3.0-safe (fix #5a)

    teams_map = {team: confed for confed, teams in _CONFED_DICT.items() for team in teams}
    df["confed_text"]       = df["team"].map(teams_map).fillna("OTHER")
    df["rival_confed_text"] = df["opponent"].map(teams_map).fillna("OTHER")
    df["confed"]       = df["confed_text"].map(_CONFED_NUMERIC).astype(int)
    df["rival_confed"] = df["rival_confed_text"].map(_CONFED_NUMERIC).astype(int)

    df["tournament_num"] = 1
    df.loc[df["tournament"].isin(["FIFA World Cup", "Copa América", "UEFA Euro", "African Cup of Nations"]), "tournament_num"] = 5
    df.loc[df["tournament"].isin(["UEFA Nations League", "CONMEBOL-UEFA Cup of Champions", "FIFA World Cup qualification", "AFC Asian Cup", "Gold Cup"]), "tournament_num"] = 4
    df.loc[df["tournament"].isin(["UEFA Euro qualification", "AFC Asian Cup qualification", "CONCACAF Nations League", "African Cup of Nations qualification", "Gold Cup qualification"]), "tournament_num"] = 3
    df.loc[df["tournament"].isin(["Friendly", "FIFA series", "CONCACAF Nations League qualification"]), "tournament_num"] = 2

    return df


def _apply_pca_merge_decay(df, *, scaler=None, pca=None, fit_transformers=True):
    """Fit or apply scaler+PCA, merge rival stats, add time-decay weight.
    fit_transformers=False requires non-None scaler and pca.
    Returns (df_engineered, scaler, pca)."""
    if not fit_transformers:
        assert scaler is not None and pca is not None, \
            "scaler and pca must be provided when fit_transformers=False"

    no_nan_rows = df[_COLUMNS_PCA].notna().all(axis=1)

    if fit_transformers:
        scaler = StandardScaler()
        pca    = PCA(n_components=2, random_state=42)
        scaled = scaler.fit_transform(df.loc[no_nan_rows, _COLUMNS_PCA])
        comps  = pca.fit_transform(scaled)
    else:
        scaled = scaler.transform(df.loc[no_nan_rows, _COLUMNS_PCA])
        comps  = pca.transform(scaled)

    df["PCA_1"] = np.nan
    df["PCA_2"] = np.nan
    df.loc[no_nan_rows, "PCA_1"] = comps[:, 0]
    df.loc[no_nan_rows, "PCA_2"] = comps[:, 1]

    _old = ["rival_gf_prom_5", "rival_gc_prom_5", "rival_elo_prom_5",
            "rival_gf_prom_15", "rival_gc_prom_15", "rival_PCA_1", "rival_PCA_2"]
    df.drop(columns=[c for c in _old if c in df.columns], inplace=True, errors='ignore')
    rival = df[["date", "team", "gf_prom_5", "gc_prom_5", "elo_prom_5",
                "gf_prom_15", "gc_prom_15", "PCA_1", "PCA_2"]].copy()
    rival.columns = ["date", "opponent", "rival_gf_prom_5", "rival_gc_prom_5", "rival_elo_prom_5",
                     "rival_gf_prom_15", "rival_gc_prom_15", "rival_PCA_1", "rival_PCA_2"]
    df = pd.merge(df, rival, on=["date", "opponent"], how="left")

    ref = pd.to_datetime("2024-06-01")
    df["days_away"]   = (ref - df["date"]).dt.days.clip(lower=0)
    df["date_weight"] = np.exp(-np.log(2) / 1277 * df["days_away"])

    return df, scaler, pca


def engineer_features(df_historia, *, scaler=None, pca=None, fit_transformers=True):
    """Full pipeline: rolling windows → PCA → rival merge → time-decay.
    For no-leakage validation: call _add_rolling_and_style on full history, split,
    then call _apply_pca_merge_decay on each split independently.
    Returns (df_engineered, scaler, pca, FEATURES)."""
    df = _add_rolling_and_style(df_historia.copy())
    df, scaler, pca = _apply_pca_merge_decay(
        df, scaler=scaler, pca=pca, fit_transformers=fit_transformers)
    return df, scaler, pca, FEATURES


# ─── Dixon-Coles rho estimation ──────────────────────────────────────────────

def estimate_rho(model, df_train: pd.DataFrame) -> float:
    """Estimate Dixon-Coles rho via MLE on training data.

    DC corrects P(0-0), P(1-0), P(0-1), P(1-1) to capture the positive
    correlation between goals scored by each team (draws more likely than
    independent Poisson predicts). rho < 0 → more draws.
    """
    xg_pred = np.maximum(model.predict(df_train[FEATURES]).astype(float), 1e-6)
    df_p = df_train.copy()
    df_p["xg_pred"] = xg_pred

    df_A = df_p[df_p["team"] < df_p["opponent"]].copy()
    df_B = (df_p[df_p["team"] > df_p["opponent"]]
            [["date", "team", "opponent", "xg_pred"]]
            .rename(columns={"team": "opponent", "opponent": "team", "xg_pred": "xg_away"}))

    matches = pd.merge(
        df_A[["date", "team", "opponent", "goals", "goals_conceded", "xg_pred"]]
            .rename(columns={"xg_pred": "xg_home"}),
        df_B, on=["date", "team", "opponent"], how="inner",
    )
    low = matches[(matches["goals"] <= 1) & (matches["goals_conceded"] <= 1)].copy()

    lams = low["xg_home"].values
    mus  = low["xg_away"].values
    xs   = low["goals"].astype(int).values
    ys   = low["goals_conceded"].astype(int).values

    def neg_log_tau(rho):
        tau = np.ones(len(low))
        tau[(xs == 0) & (ys == 0)] -= lams[(xs == 0) & (ys == 0)] * mus[(xs == 0) & (ys == 0)] * rho
        tau[(xs == 1) & (ys == 0)] += mus[(xs == 1) & (ys == 0)] * rho
        tau[(xs == 0) & (ys == 1)] += lams[(xs == 0) & (ys == 1)] * rho
        tau[(xs == 1) & (ys == 1)] -= rho
        if np.any(tau <= 0):
            return 1e10
        return -np.sum(np.log(tau))

    res = minimize_scalar(neg_log_tau, bounds=(-0.99, -0.001), method="bounded")
    return float(res.x)


# ─── model training ──────────────────────────────────────────────────────────

def train_model(name, use_tuned=False):
    print("1. Cargando datos...")
    df = pd.read_csv(f"data/models_csv/df_{name}.csv")
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values(by=["team", "date"], inplace=True)

    df_historia = df[df["date"] < "2026-06-11"].copy()
    df_mundial  = df[df["date"] >= "2026-06-11"].copy()

    print("2. Calculando ventanas, PCA y pesos...")
    df_eng, _, _, _ = engineer_features(df_historia, fit_transformers=True)

    print("3. Entrenando el motor XGBoost...")
    feat_means = df_eng[FEATURES].mean()
    df_eng[FEATURES] = df_eng[FEATURES].fillna(feat_means)
    df_train = df_eng.dropna(subset=[TARGET]).copy()

    params = _DEFAULT_PARAMS.copy()
    if use_tuned:
        tuning_path = os.path.join("results", f"tuning_{name}.json")
        if os.path.exists(tuning_path):
            with open(tuning_path, encoding="utf-8") as f:
                params.update(json.load(f)["best_params"])
        else:
            print(f"  ⚠️  {tuning_path} no existe, usando parámetros por defecto.")

    model = xgb.XGBRegressor(**params)
    model.fit(df_train[FEATURES], df_train[TARGET], sample_weight=df_train["date_weight"])

    os.makedirs("api/models", exist_ok=True)
    model.save_model(f"api/models/xgb_{name}.json")

    print("3b. Estimando corrección Dixon-Coles (rho)...")
    rho = estimate_rho(model, df_train)
    rho_path = f"api/models/rho_{name}.json"
    with open(rho_path, "w", encoding="utf-8") as f:
        json.dump({"variant": name, "rho": rho}, f)
    print(f"  ρ = {rho:.4f}  →  {rho_path}")

    print("4. Generando predicciones para el Mundial 2026...")

    # Foto fija: average of last _N_FOTO matches per team (fix #5c)
    df_eng.sort_values(by="date", inplace=True)
    last_n      = df_eng.groupby("team").tail(_N_FOTO)
    avg_part    = last_n.groupby("team")[_FOTO_NUMERIC].mean()
    confed_part = df_eng.groupby("team")["confed"].last()
    foto_fija_equipos = avg_part.join(confed_part).reset_index()[["team", *_FOTO_NUMERIC, "confed"]]

    _updated_path = os.path.join("data", "ai_models", "foto_fija_updated.csv")
    if os.path.exists(_updated_path):
        df_updated = pd.read_csv(_updated_path)
        cols_to_update = ['gf_prom_5', 'gc_prom_5', 'gf_prom_15', 'gc_prom_15', 'PCA_1', 'PCA_2', 'confed']
        df_updated_indexed = df_updated.set_index("team")
        foto_fija_indexed  = foto_fija_equipos.set_index("team")
        for col in cols_to_update:
            if col in df_updated_indexed.columns:
                foto_fija_indexed.loc[df_updated_indexed.index, col] = df_updated_indexed[col]
        foto_fija_equipos = foto_fija_indexed.reset_index()
        print(f"  ✅ Foto fija actualizada con resultados reales de {len(df_updated)} equipos.")

    foto_fija_rivales = foto_fija_equipos.copy()
    foto_fija_rivales.columns = ['opponent', 'rival_gf_prom_5', 'rival_gc_prom_5', 'rival_elo_prom_5',
                                  'rival_gf_prom_15', 'rival_gc_prom_15', 'rival_PCA_1', 'rival_PCA_2',
                                  'rival_confed']

    jornadas = {
        "J1": df_mundial[df_mundial["date"] < "2026-06-18"].copy(),
        "J2": df_mundial[(df_mundial["date"] >= "2026-06-18") & (df_mundial["date"] < "2026-06-24")].copy(),
        "J3": df_mundial[df_mundial["date"] >= "2026-06-24"].copy(),
    }

    for nombre_jornada, df_jornada in jornadas.items():
        if len(df_jornada) == 0:
            continue

        cols_a_borrar = ([c for c in foto_fija_equipos.columns if c != 'team'] +
                         [c for c in foto_fija_rivales.columns if c != 'opponent'])
        df_jornada.drop(columns=[c for c in cols_a_borrar if c in df_jornada.columns],
                        inplace=True, errors='ignore')

        df_jornada = pd.merge(df_jornada, foto_fija_equipos, on="team", how="left")
        df_jornada = pd.merge(df_jornada, foto_fija_rivales, on="opponent", how="left")
        df_jornada["tournament_num"] = 5
        df_jornada[FEATURES] = df_jornada[FEATURES].fillna(feat_means)

        df_jornada['xg_estimated'] = model.predict(df_jornada[FEATURES]).round(2)

        cols_export = ["date", "team", "opponent", "xg_estimated"]
        df_jornada.sort_values(by=["date", "team"], inplace=True)
        df_jornada[cols_export].to_csv(f"data/ai_models/xg_preds_{nombre_jornada}_{name}.csv", index=False)
        df_jornada.to_csv(f"data/ai_models/xg_preds_{nombre_jornada}_{name}_complete.csv", index=False)
        print(f"✅ Predicciones para {nombre_jornada} guardadas con éxito.")

    print("\n¡Simulación completa! Revisa la carpeta 'data'.")
    return model


# ─── hyperparameter tuning (opt-in) ──────────────────────────────────────────

def tune_hyperparameters(name, n_iter=20, subsample_frac=1.0):
    """Reactivate RandomizedSearchCV and log results to results/tuning_<name>.json.
    Does NOT change production models. Adopt with train_model(name, use_tuned=True)."""
    print(f"\n[{name}] Buscando hiperparámetros ({n_iter} iteraciones × 5 folds)...")

    df = pd.read_csv(f"data/models_csv/df_{name}.csv")
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values(by=["team", "date"], inplace=True)
    df_historia = df[df["date"] < "2026-06-11"].copy()

    df_eng, _, _, _ = engineer_features(df_historia, fit_transformers=True)
    df_eng[FEATURES] = df_eng[FEATURES].fillna(df_eng[FEATURES].mean())
    df_train = df_eng.dropna(subset=[TARGET]).copy()

    if subsample_frac < 1.0:
        df_train = df_train.tail(int(len(df_train) * subsample_frac)).copy()

    tscv = TimeSeriesSplit(n_splits=5)
    param_distributions = {
        'max_depth':              [3, 4, 5],
        'learning_rate':          [0.01, 0.05, 0.1],
        'n_estimators':           [100, 300, 500],
        'subsample':              [0.7, 0.8, 1.0],
        'tweedie_variance_power': [1.3, 1.5, 1.7],
        'colsample_bytree':       [0.7, 0.8, 1.0],
    }

    optimizador = RandomizedSearchCV(
        estimator=xgb.XGBRegressor(objective='reg:tweedie', random_state=42),
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=tscv,
        scoring='neg_mean_absolute_error',
        random_state=42,
        n_jobs=-1,
    )
    optimizador.fit(df_train[FEATURES], df_train[TARGET],
                    sample_weight=df_train["date_weight"])

    best = optimizador.best_estimator_
    result = {
        "variant":    name,
        "best_params": {
            k: (int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v)
            for k, v in optimizador.best_params_.items()
        },
        "best_cv_mae": round(-float(optimizador.best_score_), 4),
        "cv":          "TimeSeriesSplit(5)",
        "scoring":     "neg_mean_absolute_error",
        "n_iter":      n_iter,
        "feature_importances": dict(zip(FEATURES, [round(float(v), 6) for v in best.feature_importances_])),
    }

    os.makedirs("results", exist_ok=True)
    out = os.path.join("results", f"tuning_{name}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"  Mejores parámetros: {result['best_params']}")
    print(f"  CV MAE: {result['best_cv_mae']:.4f}")
    print(f"  ✅ {out}")
    return result
