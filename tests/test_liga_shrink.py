"""Encogimiento de ventanas hacia un prior basado en ELO (equipos con poca historia)."""
import numpy as np
import pandas as pd
import pytest

import xg_preds as xp
from liga import ingest, model as lm
from liga_synth import make_seasons


@pytest.fixture(scope="module")
def world():
    seasons = make_seasons()
    long = ingest.to_long(ingest.add_elo(seasons))
    feat = xp._add_rolling_and_style(lm._sorted(long).copy())
    train_seasons = sorted(feat["season"].unique())[1:]
    pri = lm.fit_priors(feat[feat["season"].isin(train_seasons)])
    return dict(seasons=seasons, long=long, feat=feat, pri=pri, train_seasons=train_seasons,
                elo_now=ingest.final_elos(seasons))


GOAL_COLS = ["gf_prom_5", "gc_prom_5", "gf_prom_15", "gc_prom_15"]
ALL_COLS = GOAL_COLS + ["elo_prom_5", "pace_20", "gf_ratio_20", "clean_20", "underdog_20"]


def test_none_and_k0_leave_every_feature_untouched(world):
    base = world["feat"]
    for sh in (None, ("const", 0), ("fill", 0)):
        out = lm.apply_shrink(base.copy(), world["pri"], sh)
        pd.testing.assert_frame_equal(out[ALL_COLS], base[ALL_COLS])
        assert "n_prev" in out.columns


def test_n_prev_counts_previous_matches_of_the_team(world):
    out = lm.apply_shrink(world["feat"].copy(), world["pri"], None)
    for _, g in out.groupby("team"):
        assert list(g["n_prev"]) == list(range(len(g)))


@pytest.mark.parametrize("mode,k", [("const", 3), ("fill", 5)])
def test_team_without_history_gets_exactly_the_elo_prior(world, mode, k):
    out = lm.apply_shrink(world["feat"].copy(), world["pri"], (mode, k))
    first = out[out["n_prev"] == 0]
    a, b = world["pri"]["gf"]
    assert np.allclose(first["gf_prom_5"], a + b * first["elo"])
    a, b = world["pri"]["gc"]
    assert np.allclose(first["gc_prom_15"], a + b * first["elo"])
    assert np.allclose(first["elo_prom_5"], world["pri"]["elo_prom_5"])


def test_fill_mode_changes_nothing_once_the_window_is_full(world):
    base, out = world["feat"], lm.apply_shrink(world["feat"].copy(), world["pri"], ("fill", 5))
    full = out["n_prev"] >= 20                      # ventana más larga (estilo, 20)
    pd.testing.assert_frame_equal(out.loc[full, ALL_COLS], base.loc[full, ALL_COLS])
    assert not np.allclose(out.loc[out["n_prev"].between(1, 4), "gf_prom_5"],
                           base.loc[out["n_prev"].between(1, 4), "gf_prom_5"])


def test_const_mode_is_a_convex_combination_of_own_mean_and_prior(world):
    base, out = world["feat"], lm.apply_shrink(world["feat"].copy(), world["pri"], ("const", 3))
    a, b = world["pri"]["gf"]
    prior = a + b * out["elo"]
    m = out["n_prev"] > 0
    lo, hi = np.minimum(base.loc[m, "gf_prom_5"], prior[m]) - 1e-9, np.maximum(base.loc[m, "gf_prom_5"], prior[m]) + 1e-9
    assert ((out.loc[m, "gf_prom_5"] >= lo) & (out.loc[m, "gf_prom_5"] <= hi)).all()


def test_huge_k_collapses_to_the_prior(world):
    out = lm.apply_shrink(world["feat"].copy(), world["pri"], ("const", 1e9))
    a, b = world["pri"]["gf"]
    assert np.allclose(out["gf_prom_5"], a + b * out["elo"], atol=1e-3)


def test_priors_are_fitted_with_training_rows_only(world):
    feat, pri = world["feat"], world["pri"]
    last = world["train_seasons"][-1]
    pri_without_last = lm.fit_priors(feat[feat["season"].isin(world["train_seasons"][:-1])])
    assert pri_without_last != pri                      # depende de las filas que se le pasan
    m = lm.train_liga(world["long"], shrink=("const", 3))
    assert m.priors == pri
    assert m.shrink == ("const", 3) and last in m.train_seasons


def test_fit_priors_gives_sensible_elo_slope(world):
    mean_elo = world["feat"]["elo"].mean()
    for key in ("gf", "gc"):
        a, b = world["pri"][key]
        assert 0.8 < a + b * mean_elo < 2.2            # goles esperados de un equipo medio
    assert world["pri"]["gf"][1] > 0 > world["pri"]["gc"][1]     # más ELO: más goles a favor, menos en contra


def test_model_with_shrink_predicts_pending_fixtures_without_nans(world):
    m = lm.train_liga(world["long"], shrink=("fill", 5))
    st = lm.team_state(world["long"], m, world["elo_now"])
    last = list(world["seasons"])[-1]
    out = lm.predict_fixtures(m, st, ingest.pending_fixtures(world["seasons"][last]))
    assert not out[["xg_home", "xg_away"]].isna().any().any() and (out[["xg_home", "xg_away"]] > 0).all().all()


def test_default_predictions_are_identical_with_and_without_explicit_none(world):
    a = lm.train_liga(world["long"])
    b = lm.train_liga(world["long"], shrink=None)
    sa, sb = lm.team_state(world["long"], a, world["elo_now"]), lm.team_state(world["long"], b, world["elo_now"])
    pd.testing.assert_frame_equal(sa, sb)


def test_holdout_reports_affected_rows_and_the_shrink_used(world):
    # un "ascendido": T05 desaparece desde la 4.ª temporada y entra NEW (sin historia previa)
    seasons = {c: (df.replace({"T05": "NEW"}) if i >= 3 else df) for i, (c, df) in enumerate(world["seasons"].items())}
    long = ingest.to_long(ingest.add_elo(seasons))
    v = lm.holdout_validation(long, shrink=("fill", 5))
    assert v["shrink"] == ["fill", 5]
    aff = v["affected"]
    assert aff["n"] > 0 and {"model", "baseline_constant"} <= set(aff)
    assert aff["n"] < v["n_test_rows"]
