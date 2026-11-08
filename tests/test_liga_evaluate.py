"""Puntuación de predicciones de temporada: RPS de posiciones y log-loss de zonas."""
import math

import pytest

from liga import evaluate as ev


def test_rps_is_zero_for_a_perfect_prediction():
    assert ev.rps([0, 1, 0, 0], 2) == 0.0


def test_rps_of_uniform_prediction_matches_hand_computation():
    # n=4, real 1.º: cdf pred [.25,.5,.75], real [1,1,1] → (.5625+.25+.0625)/3
    assert ev.rps([0.25] * 4, 1) == pytest.approx(0.875 / 3)


def test_rps_penalises_distance_not_just_misses():
    near = ev.rps([0, 1, 0, 0], 1)     # acierta por 1 puesto de diferencia
    far = ev.rps([0, 0, 0, 1], 1)      # falla por 3 puestos
    assert 0 < near < far


def test_log_loss_binary():
    assert ev.log_loss_binary(0.5, 1) == pytest.approx(math.log(2))
    assert ev.log_loss_binary(1.0, 1) == pytest.approx(0.0, abs=1e-5)
    assert ev.log_loss_binary(0.0, 1) > 10                # acotado por eps, no infinito
    assert ev.log_loss_binary(0.0, 0) == pytest.approx(0.0, abs=1e-5)


def _teams(dists):
    return {name: {"position_dist": d, "p_champion": d[0], "p_champions": sum(d[:2]), "p_relegation": d[-1]}
            for name, d in dists.items()}


def test_score_season_on_a_toy_league():
    teams = _teams({"A": [1, 0, 0], "B": [0, 1, 0], "C": [0, 0, 1]})
    actual = {"A": 1, "B": 2, "C": 3}
    s = ev.score_season(teams, actual, champions=2, relegation=1)
    assert s["rps"] == 0.0 and s["logloss_champion"] == pytest.approx(0.0, abs=1e-5)
    assert s["coverage80"] == 1.0
    bad = ev.score_season(teams, {"A": 3, "B": 2, "C": 1}, champions=2, relegation=1)
    assert bad["rps"] > 0.5 and bad["logloss_champion"] > 1.0 and bad["coverage80"] < 1.0


def test_score_season_rejects_mismatched_teams():
    with pytest.raises(ValueError, match="equipos"):
        ev.score_season(_teams({"A": [1, 0]}), {"A": 1, "B": 2}, champions=1, relegation=1)
