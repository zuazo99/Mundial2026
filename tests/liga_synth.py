"""Liga sintética de fuerzas conocidas, compartida por los tests de modelo y simulación."""
import itertools  # noqa: F401

import numpy as np
import pandas as pd

N_TEAMS, N_SEASONS = 12, 5
TEAMS = [f"T{i:02d}" for i in range(N_TEAMS)]
STRENGTH = dict(zip(TEAMS, np.linspace(-1.0, 1.0, N_TEAMS)))  # T11 fuerte, T00 débil


def _round_robin_rounds(teams):
    """Método del círculo: n-1 rondas, cada equipo juega exactamente una vez por ronda."""
    ts = list(teams)
    n = len(ts)
    rounds = []
    for _ in range(n - 1):
        rounds.append([(ts[i], ts[n - 1 - i]) for i in range(n // 2)])
        ts = [ts[0]] + [ts[-1]] + ts[1:-1]
    return rounds


def make_seasons(seed=0, last_fraction=0.6):
    """Doble vuelta semanal con calendario válido; la última temporada solo se ha jugado en parte."""
    rng = np.random.default_rng(seed)
    seasons = {}
    for k in range(N_SEASONS):
        first_leg = [[(a, h) if rng.random() < 0.5 else (h, a) for h, a in r]   # alterna localías de la ida
                     for r in _round_robin_rounds(TEAMS)]
        rounds = first_leg + [[(a, h) for h, a in r] for r in first_leg]       # la vuelta es el espejo exacto
        start = pd.Timestamp(2021 + k, 8, 15)
        rows = []
        for ri, matches in enumerate(rounds):
            for h, a in matches:
                d = STRENGTH[h] - STRENGTH[a]
                lam_h, lam_a = 1.45 * np.exp(0.55 * d) * 1.15, 1.45 * np.exp(-0.55 * d) / 1.1
                rows.append((start + pd.Timedelta(days=7 * ri), h, a,
                             int(rng.poisson(lam_h)), int(rng.poisson(lam_a))))
        df = pd.DataFrame(rows, columns=["date", "home", "away", "hg", "ag"])
        if k == N_SEASONS - 1:
            df = df.iloc[: int(len(df) * last_fraction)]
        seasons[f"{21 + k}{22 + k}"] = df.reset_index(drop=True)
    return seasons
