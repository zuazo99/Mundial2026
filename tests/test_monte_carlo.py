"""Prueba de humo del Monte Carlo vectorizado y consistencia de datos."""
import numpy as np
import pytest

import monte_carlo as mc


def test_poisson_pmf_sums_to_one():
    assert mc._poisson_pmf(1.3).sum() == pytest.approx(1.0, abs=1e-3)


def test_dc_sample_returns_nonnegative_ints():
    np.random.seed(0)
    for rho in (0.0, -0.05):
        for _ in range(100):
            g1, g2 = mc._dc_sample(1.4, 0.9, rho)
            assert g1 >= 0 and g2 >= 0


def test_knockout_match_never_ends_level():
    np.random.seed(0)
    for _ in range(200):
        g1, g2 = mc.simulate_match(1.0, 1.0, ko=True)
        assert g1 != g2


def test_sort_group_orders_by_points_then_gd_then_gf():
    stats = [dict(pts=3, gd=0, gf=2), dict(pts=6, gd=-1, gf=1), dict(pts=3, gd=2, gf=2)]
    assert [s["pts"] for s in mc.sort_group(stats)] == [6, 3, 3]
    assert mc.sort_group(stats)[1]["gd"] == 2


def test_groups_are_twelve_by_four_unique_teams():
    teams = [t for g in mc.GROUPS for t in g[1:]]  # g[0] es la letra
    assert len(mc.GROUPS) == 12 and len(teams) == 48 and len(set(teams)) == 48


def test_features_in_sync_with_training():
    import xg_preds
    assert mc.FEATURES == xg_preds.FEATURES
