"""Backtest de la fuente de histórico: ¿calentar el ELO desde 1993 mejora al modelo?

Variantes (σ = 0, sin encogimiento; mismas jornadas de corte 7/14/21 y misma puntuación que liga_backtest.py):
  A  : ELO desde 2016/17 (arranque en frío), entrena desde 2017/18  — comportamiento de producción.
  B1 : ELO desde 1993/94, mismas filas de entrenamiento que A (desde 2017/18) — aísla el efecto del ELO.
  B2 : ELO desde 1993/94, entrena también con 2016/17 (ya no es de arranque en frío).

Regla de decisión fijada ANTES de ejecutar: B se adopta si, frente a A, ninguna de [RPS, log-loss de campeón,
de Champions, de descenso] empeora más de 0,0005 (ruido) en la selección (2019/20-2022/23) NI en la confirmación
(2023/24 y 2024/25). Escribe results/liga_backtest_elo.csv.

    python scripts/liga_backtest_elo.py [directorio_raw_hist]
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import numpy as np
import pandas as pd
from config import path
from liga import evaluate as ev, ingest, model as lm, simulate as sim

RAW = sys.argv[1] if len(sys.argv) > 1 else path("data", "liga", "raw_hist")
SEASONS = ["1920", "2021", "2122", "2223", "2324", "2425"]
SELECTION, CONFIRMATION = ["1920", "2021", "2122", "2223"], ["2324", "2425"]
CUTS = {7: 70, 14: 140, 21: 210}
VARIANTS = {"A": dict(chain_from="1617", train_from=None),
            "B1": dict(chain_from=None, train_from="1718"),
            "B2": dict(chain_from=None, train_from="1617")}
N_SIMS, TOL = 2000, 0.0005
METRICS = ["rps", "logloss_champion", "logloss_top", "logloss_relegation"]

sp1, sp2_all = ingest.load_seasons(RAW, "SP1"), ingest.load_seasons(RAW, "SP2")
rows, t0 = [], time.time()
for S in SEASONS:
    full = sp1[S]
    teams = sorted(set(full.home) | set(full.away))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    hg, ag = np.zeros((n, n), int), np.zeros((n, n), int)
    for r in full.itertuples():
        hg[idx[r.home], idx[r.away]], ag[idx[r.home], idx[r.away]] = r.hg, r.ag
    actual = {teams[i]: k + 1 for k, i in enumerate(sim.final_order(hg, ag))}
    for md, cut in CUTS.items():
        for vname, v in VARIANTS.items():
            lo = ingest.season_key(v["chain_from"]) if v["chain_from"] else 0
            seasons = {c: (df if c != S else df.sort_values("date", kind="stable").head(cut))
                       for c, df in sp1.items() if lo <= ingest.season_key(c) <= ingest.season_key(S)}
            second = {c: df for c, df in sp2_all.items() if lo <= ingest.season_key(c) < ingest.season_key(S)}
            long = ingest.to_long(ingest.add_elo(seasons, second_div=second))
            elo_now = ingest.final_elos(seasons, second_div=second)
            m = lm.train_liga(long, train_from=v["train_from"])
            xg = lm.predict_fixtures(m, lm.team_state(long, m, elo_now), ingest.pending_fixtures(seasons[S]))
            out = sim.simulate_season(teams, seasons[S][["home", "away", "hg", "ag"]], xg, m.rho, N_SIMS, 1)["teams"]
            rows.append(dict(season=S, md=md, variant=vname, **ev.score_season(out, actual)))
        print(f"{S} md{md} listo ({time.time() - t0:.0f}s)", flush=True)
df = pd.DataFrame(rows)
df.to_csv(path("results", "liga_backtest_elo.csv"), index=False)

print("\n=== Medias por variante ===")
for name, seas in (("selección", SELECTION), ("confirmación", CONFIRMATION)):
    m = df[df.season.isin(seas)].groupby("variant")[METRICS].mean()
    print(f"\n{name} ({', '.join(seas)}):\n{m.round(4).to_string()}")
    for b in ("B1", "B2"):
        worse = {k: round(float(m.loc[b, k] - m.loc["A", k]), 4) for k in METRICS if m.loc[b, k] - m.loc["A", k] > TOL}
        print(f"  {b} vs A: " + ("empeora " + str(worse) if worse else "no empeora"))
print("FIN", flush=True)
