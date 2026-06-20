"""
Temporal hold-out validation and calibration for the XG model.

Loads raw match history, splits at a cutoff date, fits PCA/scaler on
train-only (no leakage), trains XGBoost on train, predicts on test, and
reports regression metrics + 1X2 calibration + per-confederation breakdown.

Usage (from repo root):
    python3 src/validacion.py                            # all 3 variants, 2025-09-16
    python3 src/validacion.py misterclaude               # single variant
    python3 src/validacion.py misterclaude 2025-06-16    # earlier cutoff

Output: results/validation_<variant>.json
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import poisson as scipy_poisson
from sklearn.calibration import calibration_curve
from sklearn.metrics import (mean_absolute_error,
                             mean_poisson_deviance, mean_squared_error)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import xgboost as xgb
from xg_preds import (FEATURES, TARGET, _CONFED_NUMERIC, _DEFAULT_PARAMS,
                       _add_rolling_and_style, _apply_pca_merge_decay, estimate_rho)

VARIANTS = ["misterclaude", "gemaldini", "dav_gpo"]


def _poisson_1x2(xg_home: float, xg_away: float, max_goals: int = 10):
    """Analytical 1X2 probs from independent Poisson (mirrors MatchCard.astro logic)."""
    ks = np.arange(0, max_goals + 1)
    pmf_h = scipy_poisson.pmf(ks, max(xg_home, 1e-9))
    pmf_a = scipy_poisson.pmf(ks, max(xg_away, 1e-9))
    joint  = np.outer(pmf_h, pmf_a)
    p_home = float(np.tril(joint, -1).sum())
    p_draw = float(np.trace(joint))
    p_away = max(0.0, 1.0 - p_home - p_draw)
    return p_home, p_draw, p_away


def _dc_1x2(xg_home: float, xg_away: float, rho: float, max_goals: int = 10):
    """Dixon-Coles corrected 1X2 probabilities.

    Applies the τ multiplier to cells (0-0), (1-0), (0-1), (1-1) to capture
    the positive correlation between each team's goals (rho < 0 → more draws).
    After correction the matrix is renormalized so probabilities still sum to 1.
    """
    lam = max(xg_home, 1e-9)
    mu  = max(xg_away, 1e-9)
    ks = np.arange(0, max_goals + 1)
    joint = np.outer(scipy_poisson.pmf(ks, lam), scipy_poisson.pmf(ks, mu))
    # τ corrections for low-scoring cells
    joint[0, 0] *= max(0.0, 1.0 - lam * mu * rho)
    joint[1, 0] *= max(0.0, 1.0 + mu * rho)
    joint[0, 1] *= max(0.0, 1.0 + lam * rho)
    joint[1, 1] *= max(0.0, 1.0 - rho)
    joint  = np.maximum(joint, 0.0)
    joint /= joint.sum()
    p_home = float(np.tril(joint, -1).sum())
    p_draw = float(np.trace(joint))
    p_away = max(0.0, 1.0 - p_home - p_draw)
    return p_home, p_draw, p_away


def _elo_1x2(elo_home: float, elo_away: float, draw_rate: float):
    """ELO-only 1X2 baseline with empirical draw mass."""
    p_win_raw = 1.0 / (1.0 + 10.0 ** ((elo_away - elo_home) / 400.0))
    return (p_win_raw * (1.0 - draw_rate),
            draw_rate,
            (1.0 - p_win_raw) * (1.0 - draw_rate))


def validate_model(name: str, cutoff: str = "2025-09-16", params: dict = None) -> dict:
    cutoff_dt = pd.to_datetime(cutoff)
    params = params or _DEFAULT_PARAMS
    print(f"\n{'='*60}")
    print(f"[{name}] Validación — corte: {cutoff}")

    # 1. Load and sort
    df = pd.read_csv(os.path.join(ROOT, "data", "models_csv", f"df_{name}.csv"))
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values(by=["team", "date"], inplace=True)
    df_historia = df[df["date"] < "2026-06-11"].copy()

    # 2. Rolling features on FULL history so early test rows keep their pre-cutoff window context
    df_historia = _add_rolling_and_style(df_historia)

    # 3. Temporal split
    df_train_raw = df_historia[df_historia["date"] < cutoff_dt].copy()
    df_test_raw  = df_historia[df_historia["date"] >= cutoff_dt].copy()

    if len(df_test_raw) < 100:
        print(f"  ⚠️  Solo {len(df_test_raw)} filas de test — corte demasiado tardío.")
        return {}

    print(f"  Train: {len(df_train_raw):,} filas  |  Test: {len(df_test_raw):,} filas")

    # 4. Fit PCA/scaler on TRAIN only, transform both (no leakage)
    df_train, scaler, pca = _apply_pca_merge_decay(df_train_raw, fit_transformers=True)
    df_test, _, _         = _apply_pca_merge_decay(df_test_raw, scaler=scaler, pca=pca,
                                                    fit_transformers=False)

    # 5. NaN fill using train means, drop rows with missing target
    feat_means = df_train[FEATURES].mean()
    df_train[FEATURES] = df_train[FEATURES].fillna(feat_means)
    df_test[FEATURES]  = df_test[FEATURES].fillna(feat_means)
    df_train = df_train.dropna(subset=[TARGET]).copy()
    df_test  = df_test.dropna(subset=[TARGET]).copy()

    # 6. Train
    model = xgb.XGBRegressor(**params)
    model.fit(df_train[FEATURES], df_train[TARGET],
              sample_weight=df_train["date_weight"])

    # 6b. Estimate Dixon-Coles rho from train only (no leakage)
    rho = estimate_rho(model, df_train)
    print(f"  Dixon-Coles ρ (train): {rho:.4f}")

    # 7. Predict
    y_true = df_test[TARGET].values.astype(float)
    y_pred = np.maximum(model.predict(df_test[FEATURES]).astype(float), 1e-6)

    # ─── Regression metrics ────────────────────────────────────────────────
    baseline = np.full_like(y_true, float(df_train[TARGET].mean()))
    reg = {
        "mae":              round(float(mean_absolute_error(y_true, y_pred)), 4),
        "rmse":             round(float(mean_squared_error(y_true, y_pred) ** 0.5), 4),
        "poisson_deviance": round(float(mean_poisson_deviance(y_true, y_pred)), 4),
    }
    reg_base = {
        "mae":              round(float(mean_absolute_error(y_true, baseline)), 4),
        "rmse":             round(float(mean_squared_error(y_true, baseline) ** 0.5), 4),
        "poisson_deviance": round(float(mean_poisson_deviance(y_true, np.maximum(baseline, 1e-6))), 4),
    }

    # ─── 1X2 reconstruction ───────────────────────────────────────────────
    # Canonical match orientation: team < opponent alphabetically (one row per match)
    df_p = df_test.copy()
    df_p["xg_pred"] = y_pred

    df_A = df_p[df_p["team"] < df_p["opponent"]].copy()
    df_B = df_p[df_p["team"] > df_p["opponent"]][["date", "team", "opponent", "xg_pred"]].copy()
    df_B = df_B.rename(columns={"team": "opponent", "opponent": "team", "xg_pred": "xg_away"})

    cols_A = ["date", "team", "opponent", "goals", "goals_conceded", "elo", "opponent_elo", "xg_pred"]
    matches = pd.merge(
        df_A[[c for c in cols_A if c in df_A.columns]].rename(columns={"xg_pred": "xg_home"}),
        df_B,
        on=["date", "team", "opponent"],
        how="inner",
    )

    n_matches = len(matches)
    print(f"  Partidos 1X2 reconstruidos: {n_matches}")
    if n_matches < 50:
        print("  ⚠️  Pocos partidos emparejados — métricas 1X2 poco fiables.")

    true_hw = (matches["goals"] > matches["goals_conceded"]).astype(int).values
    true_d  = (matches["goals"] == matches["goals_conceded"]).astype(int).values
    true_aw = (matches["goals"] < matches["goals_conceded"]).astype(int).values
    onehot  = np.column_stack([true_hw, true_d, true_aw]).astype(float)

    # Poisson (independent) probabilities
    proba_rows = [_poisson_1x2(r.xg_home, r.xg_away) for r in matches.itertuples()]
    proba = np.clip(np.array(proba_rows), 1e-9, 1.0)

    # Dixon-Coles corrected probabilities
    proba_dc_rows = [_dc_1x2(r.xg_home, r.xg_away, rho) for r in matches.itertuples()]
    proba_dc = np.clip(np.array(proba_dc_rows), 1e-9, 1.0)

    draw_rate = float(true_d.mean())
    proba_elo = np.clip(
        np.array([_elo_1x2(r.elo, r.opponent_elo, draw_rate) for r in matches.itertuples()]),
        1e-9, 1.0,
    )

    def _brier(p): return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))
    def _logloss(p):
        tp = np.where(true_hw, p[:, 0], np.where(true_d, p[:, 1], p[:, 2]))
        return -float(np.mean(np.log(np.maximum(tp, 1e-9))))

    brier_poisson = _brier(proba)
    brier_dc      = _brier(proba_dc)
    brier_elo     = _brier(proba_elo)
    brier_uniform = _brier(np.full_like(onehot, 1/3))

    ll_poisson = _logloss(proba)
    ll_dc      = _logloss(proba_dc)
    ll_elo     = _logloss(proba_elo)

    # Calibration on DC probs (our best model)
    prob_pred_cal, prob_true_cal = calibration_curve(
        true_hw, proba_dc[:, 0], n_bins=10, strategy="quantile")
    bin_counts = np.bincount(
        np.searchsorted(prob_pred_cal, proba_dc[:, 0]).clip(0, len(prob_pred_cal) - 1),
        minlength=len(prob_pred_cal))
    ece_dc = float(np.sum(bin_counts / max(n_matches, 1) * np.abs(prob_true_cal - prob_pred_cal)))

    # ECE on plain Poisson for comparison
    prob_pred_p, prob_true_p = calibration_curve(
        true_hw, proba[:, 0], n_bins=10, strategy="quantile")
    bin_counts_p = np.bincount(
        np.searchsorted(prob_pred_p, proba[:, 0]).clip(0, len(prob_pred_p) - 1),
        minlength=len(prob_pred_p))
    ece_poisson = float(np.sum(bin_counts_p / max(n_matches, 1) * np.abs(prob_true_p - prob_pred_p)))

    # ─── Per-confederation breakdown ──────────────────────────────────────
    confed_inv = {v: k for k, v in _CONFED_NUMERIC.items()}
    per_confed = {}
    for code, cname in confed_inv.items():
        mask = (df_p["confed"].values == code)
        if mask.sum() < 10:
            continue
        per_confed[cname] = {
            "mae": round(float(mean_absolute_error(y_true[mask], y_pred[mask])), 4),
            "n":   int(mask.sum()),
        }

    result = {
        "variant":        name,
        "cutoff":         cutoff,
        "n_test_rows":    int(len(df_test)),
        "n_test_matches": int(n_matches),
        "dixon_coles_rho": round(rho, 4),
        "regression": {
            "model":               reg,
            "baseline_mean_goals": reg_base,
        },
        "match_1x2": {
            "poisson":         {"brier": round(brier_poisson, 4), "log_loss": round(ll_poisson, 4), "ece": round(ece_poisson, 4)},
            "dixon_coles":     {"brier": round(brier_dc,      4), "log_loss": round(ll_dc,      4), "ece": round(ece_dc,      4)},
            "baseline_elo":    {"brier": round(brier_elo,     4), "log_loss": round(ll_elo,     4)},
            "baseline_uniform":{"brier": round(brier_uniform, 4), "log_loss": round(float(np.log(3)), 4)},
            "draw_rate":       round(draw_rate, 4),
        },
        "calibration_curve_dc": {
            "prob_pred": [round(float(v), 4) for v in prob_pred_cal],
            "prob_true": [round(float(v), 4) for v in prob_true_cal],
            "strategy":  "quantile",
            "n_bins":    10,
        },
        "per_confederation": per_confed,
    }

    # ─── Print summary ────────────────────────────────────────────────────
    w = 32
    print(f"\n  {'Métrica':<{w}} {'Modelo':>10} {'Baseline':>12}")
    print(f"  {'-'*(w+24)}")
    print(f"  {'MAE goles':<{w}} {reg['mae']:>10.4f} {reg_base['mae']:>12.4f}")
    print(f"  {'RMSE goles':<{w}} {reg['rmse']:>10.4f} {reg_base['rmse']:>12.4f}")
    print(f"  {'Poisson deviance':<{w}} {reg['poisson_deviance']:>10.4f} {reg_base['poisson_deviance']:>12.4f}")
    print()
    print(f"  {'':>{w}} {'Poisson':>10} {'DC(ρ=' + f'{rho:.3f})':>12} {'ELO base':>10} {'Uniforme':>10}")
    print(f"  {'Brier 1X2':<{w}} {brier_poisson:>10.4f} {brier_dc:>12.4f} {brier_elo:>10.4f} {brier_uniform:>10.4f}")
    print(f"  {'Log-loss 1X2':<{w}} {ll_poisson:>10.4f} {ll_dc:>12.4f} {ll_elo:>10.4f} {np.log(3):>10.4f}")
    print(f"  {'ECE calibración':<{w}} {ece_poisson:>10.4f} {ece_dc:>12.4f}")
    print()
    print("  Por confederación (MAE): " +
          "  ".join(f"{k}={v['mae']:.3f}(n={v['n']})" for k, v in per_confed.items()))

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    out = os.path.join(ROOT, "results", f"validation_{name}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n  ✅ {out}")

    return result


if __name__ == "__main__":
    args       = sys.argv[1:]
    variant_in = args[0] if args else None
    cutoff_in  = args[1] if len(args) > 1 else "2025-09-16"
    for v in ([variant_in] if variant_in else VARIANTS):
        validate_model(v, cutoff_in)
