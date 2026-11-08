"""Puntuación de predicciones de temporada de La Liga.

Reglas de puntuación propias (la cobertura de intervalos se reporta, pero no se usa para decidir):
  * RPS (ranked probability score) de la distribución de posiciones frente a la posición final;
  * log-loss de las probabilidades de campeón, zona Champions y descenso.
"""
import math

import numpy as np

EPS = 1e-6


def rps(position_dist, actual_position: int) -> float:
    """RPS de una distribución de posiciones (1..n) frente a la posición real (1-based). 0 = perfecto."""
    d = np.asarray(position_dist, dtype=float)
    n = len(d)
    obs = np.zeros(n)
    obs[actual_position - 1] = 1.0
    diff = np.cumsum(d)[:-1] - np.cumsum(obs)[:-1]
    return float((diff ** 2).sum() / (n - 1))


def log_loss_binary(p: float, outcome: int) -> float:
    p = min(max(float(p), EPS), 1 - EPS)
    return -math.log(p if outcome else 1 - p)


def score_season(teams: dict, actual_positions: dict, champions: int = 4, relegation: int = 3) -> dict:
    """Puntúa la salida `teams` de simulate_season frente a la clasificación final real."""
    if set(teams) != set(actual_positions):
        raise ValueError("Los equipos de la predicción y de la clasificación real no coinciden.")
    n = len(teams)
    r, lc, lt, lr, cov = [], [], [], [], []
    for t, v in teams.items():
        pos = actual_positions[t]
        r.append(rps(v["position_dist"], pos))
        lc.append(log_loss_binary(v["p_champion"], pos == 1))
        lt.append(log_loss_binary(v["p_champions"], pos <= champions))
        lr.append(log_loss_binary(v["p_relegation"], pos > n - relegation))
        cdf = np.cumsum(v["position_dist"])
        lo, hi = int(np.argmax(cdf >= 0.10)) + 1, int(np.argmax(cdf >= 0.90)) + 1
        cov.append(lo <= pos <= hi)
    return {"rps": float(np.mean(r)), "logloss_champion": float(np.mean(lc)), "logloss_top": float(np.mean(lt)),
            "logloss_relegation": float(np.mean(lr)), "coverage80": float(np.mean(cov))}
