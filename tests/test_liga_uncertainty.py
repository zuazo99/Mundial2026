"""Incertidumbre en la fuerza de los equipos (σ) dentro de la simulación de temporada."""
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from liga import ingest, simulate as sim
from liga_synth import TEAMS, make_seasons

# Referencia de la salida de `teams` con el código anterior a la mejora (σ = 0, misma semilla)
GOLDEN_TEAMS_SHA256 = "3ce01fe586f8e5cef754bddcd7a6115e0f739e578d3cb671aee0ac9ff74f0f38"


@pytest.fixture(scope="module")
def league():
    seasons = make_seasons()
    last = list(seasons)[-1]
    pend = ingest.pending_fixtures(seasons[last])
    pend["xg_home"], pend["xg_away"] = 1.4, 1.1
    return dict(played=seasons[last][["home", "away", "hg", "ag"]], pending=pend)


def _run(league, sigma, n_sims=600, seed=11, **kw):
    return sim.simulate_season(TEAMS, league["played"], league["pending"], -0.05, n_sims, seed,
                               strength_sigma=sigma, **kw)


def test_sigma_zero_is_byte_identical_to_the_previous_implementation(league):
    out = sim.simulate_season(TEAMS, league["played"], league["pending"], -0.05, 300, 11)
    assert hashlib.sha256(json.dumps(out["teams"], sort_keys=True).encode()).hexdigest() == GOLDEN_TEAMS_SHA256
    explicit = _run(league, 0.0, n_sims=300)
    assert explicit["teams"] == out["teams"]


def test_meta_records_sigma(league):
    assert _run(league, 0.0, n_sims=50)["meta"]["strength_sigma"] == 0.0
    assert _run(league, 0.12, n_sims=50)["meta"]["strength_sigma"] == 0.12


def test_negative_sigma_is_rejected(league):
    with pytest.raises(ValueError, match="sigma"):
        _run(league, -0.1, n_sims=10)


@pytest.mark.parametrize("n_sims", [37, 450, 701])        # incluye tamaños que no son múltiplos del lote
def test_probability_invariants_hold_with_uncertainty(league, n_sims):
    t = _run(league, 0.15, n_sims=n_sims)["teams"]
    assert sum(v["p_champion"] for v in t.values()) == pytest.approx(1.0)
    assert sum(v["p_champions"] for v in t.values()) == pytest.approx(4.0)
    assert sum(v["p_relegation"] for v in t.values()) == pytest.approx(3.0)
    pos = np.array([v["position_dist"] for v in t.values()])
    assert pos.sum(axis=0) == pytest.approx(np.ones(len(TEAMS)))
    assert pos.sum(axis=1) == pytest.approx(np.ones(len(TEAMS)))


def test_same_seed_is_deterministic_and_sigma_changes_the_result(league):
    assert _run(league, 0.15, n_sims=300) == _run(league, 0.15, n_sims=300)
    assert _run(league, 0.15, n_sims=300) != _run(league, 0.0, n_sims=300)


def test_small_sigma_is_continuous_with_zero(league):
    a, b = _run(league, 0.0, n_sims=3000)["teams"], _run(league, 0.01, n_sims=3000)["teams"]
    for t in TEAMS:
        assert abs(a[t]["exp_points"] - b[t]["exp_points"]) < 1.0


def test_uncertainty_flattens_an_extreme_favourite_and_widens_positions():
    # Un equipo dominante: con σ > 0 su P(campeón) baja y la dispersión de posiciones de un equipo medio sube
    teams = [f"E{i}" for i in range(6)]
    import itertools
    pend = pd.DataFrame([(h, a, 2.6 if h == "E0" else (0.7 if a == "E0" else 1.3),
                          0.6 if h == "E0" else (2.2 if a == "E0" else 1.3))
                         for h, a in itertools.permutations(teams, 2)], columns=["home", "away", "xg_home", "xg_away"])
    played = pd.DataFrame(columns=["home", "away", "hg", "ag"])
    a = sim.simulate_season(teams, played, pend, 0.0, 4000, 3, strength_sigma=0.0)["teams"]
    b = sim.simulate_season(teams, played, pend, 0.0, 4000, 3, strength_sigma=0.25)["teams"]
    assert b["E0"]["p_champion"] < a["E0"]["p_champion"]

    def spread(v):
        d = np.array(v["position_dist"]); pos = np.arange(1, len(d) + 1)
        return float(np.sqrt((d * pos ** 2).sum() - (d * pos).sum() ** 2))
    assert spread(b["E3"]) > spread(a["E3"])


def test_matrix_sampler_matches_the_vector_sampler_statistically():
    rng = np.random.default_rng(5)
    lam, mu = np.full((4000, 3), 1.5), np.full((4000, 3), 1.0)
    hg, ag = sim._sample_matrix(lam, mu, -0.08, rng)
    assert hg.shape == (4000, 3)
    assert hg.mean() == pytest.approx(1.5, abs=0.06) and ag.mean() == pytest.approx(1.0, abs=0.05)
    d = (hg == ag).mean()
    rng2 = np.random.default_rng(6)
    h2, a2 = sim.sample_scores(np.full(3, 1.5), np.full(3, 1.0), -0.08, 4000, rng2)
    assert d == pytest.approx((h2 == a2).mean(), abs=0.03)
