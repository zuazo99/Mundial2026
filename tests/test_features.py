"""Invariantes del feature engineering (xg_preds)."""
import numpy as np
import pandas as pd
import pytest

import xg_preds as xp


def _toy_history(n=30):
    """Dos selecciones jugándose partidos alternos (2 filas por partido)."""
    rows = []
    for i in range(n):
        d = pd.Timestamp("2020-01-01") + pd.Timedelta(days=30 * i)
        g1, g2 = i % 4, (i * 3) % 3
        rows.append(dict(date=d, team="Spain", opponent="Brazil", goals=g1, goals_conceded=g2,
                         elo=2000 + i, opponent_elo=1950 + i, is_home=1, tournament="Friendly"))
        rows.append(dict(date=d, team="Brazil", opponent="Spain", goals=g2, goals_conceded=g1,
                         elo=1950 + i, opponent_elo=2000 + i, is_home=0, tournament="Friendly"))
    df = pd.DataFrame(rows)
    return df.sort_values(["team", "date"]).reset_index(drop=True)


def test_features_unique_and_expected_size():
    assert len(xp.FEATURES) == 20
    assert len(set(xp.FEATURES)) == 20


def test_rolling_windows_do_not_leak_current_match():
    df = xp._add_rolling_and_style(_toy_history())
    for _, g in df.groupby("team"):
        assert np.isnan(g["gf_prom_5"].iloc[0])  # nada previo → NaN
        # la ventana de la fila i usa solo partidos < i
        expected = g["goals"].iloc[:5].mean()
        assert g["gf_prom_5"].iloc[5] == pytest.approx(expected)


def test_engineer_features_returns_all_model_columns():
    df, scaler, pca, feats = xp.engineer_features(_toy_history())
    assert feats == xp.FEATURES
    assert set(xp.FEATURES) <= set(df.columns)
    assert scaler is not None and pca is not None


def test_date_weight_in_unit_interval_and_decays():
    df, *_ = xp.engineer_features(_toy_history())
    assert df["date_weight"].between(0, 1).all()
    old, new = df.sort_values("date").iloc[0], df.sort_values("date").iloc[-1]
    assert old["date_weight"] <= new["date_weight"]


def test_confed_and_tournament_encoding():
    df = _toy_history(3)
    df.loc[0, "tournament"] = "FIFA World Cup"
    out = xp._add_rolling_and_style(df)
    assert out.loc[out["team"] == "Spain", "confed"].iloc[0] == xp._CONFED_NUMERIC["UEFA"]
    assert out.loc[out["team"] == "Brazil", "confed"].iloc[0] == xp._CONFED_NUMERIC["CONMEBOL"]
    assert (out["tournament"].eq("FIFA World Cup") & out["tournament_num"].eq(5)).any()
    assert out.loc[out["tournament"] == "Friendly", "tournament_num"].eq(2).all()


def test_pca_transform_with_fitted_objects_is_consistent():
    df = xp._add_rolling_and_style(_toy_history())
    a, scaler, pca = xp._apply_pca_merge_decay(df.copy(), fit_transformers=True)
    b, _, _ = xp._apply_pca_merge_decay(df.copy(), scaler=scaler, pca=pca, fit_transformers=False)
    pd.testing.assert_series_equal(a["PCA_1"], b["PCA_1"])


def test_decay_parameters_default_to_world_cup_constants_and_can_be_overridden():
    h = _toy_history()
    default, *_ = xp.engineer_features(h)
    same, *_ = xp.engineer_features(h, decay_ref=xp.DECAY_REF_DATE, half_life_days=xp.DECAY_HALF_LIFE_DAYS)
    pd.testing.assert_series_equal(default["date_weight"], same["date_weight"])

    ref = h["date"].max()
    custom, *_ = xp.engineer_features(h, decay_ref=ref, half_life_days=365)
    assert custom.loc[custom["date"] == ref, "date_weight"].eq(1.0).all()
    one_year = custom.loc[custom["date"] == ref - pd.Timedelta(days=365), "date_weight"]
    if len(one_year):
        assert one_year.iloc[0] == pytest.approx(0.5, abs=0.02)
