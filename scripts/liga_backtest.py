"""Backtest de temporada de La Liga: (temporada, corte de jornada) x encogimiento x sigma.

Corta cada temporada tras la jornada 7/14/21, entrena solo con datos hasta ese punto (una vez por
temporada, corte y encogimiento), simula varios valores de sigma y puntúa frente a la clasificación final real
(RPS, log-loss de campeón/Champions/descenso, cobertura del 80 %). Escribe results/liga_backtest.csv.

    python scripts/liga_backtest.py

Resultados y decisión: ver la sección "Calidad y plausibilidad" de CLAUDE.md. Excluye 2024/25 y 2025/26
(les faltan resultados, su tabla final está incompleta).
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from config import path
import numpy as np, pandas as pd
from liga import ingest, model as lm, simulate as sim, evaluate as ev
RAW=path("data","liga","raw")
sp1=ingest.load_seasons(RAW,"SP1"); sp2_all=ingest.load_seasons(RAW,"SP2")
SEASONS=["1920","2021","2122","2223","2324"]; CUTS={7:70,14:140,21:210}
SHRINKS={"base":None,"fill":("fill",1)}; SIGMAS=[0.0,0.05,0.10,0.15,0.20]; N_SIMS=2000
rows=[]; t0=time.time()
for S in SEASONS:
    full=sp1[S]; teams=sorted(set(full.home)|set(full.away)); idx={t:i for i,t in enumerate(teams)}; n=len(teams)
    hg=np.zeros((n,n),int); ag=np.zeros((n,n),int)
    for r in full.itertuples(): hg[idx[r.home],idx[r.away]],ag[idx[r.home],idx[r.away]]=r.hg,r.ag
    order=sim.final_order(hg,ag); actual={teams[i]:k+1 for k,i in enumerate(order)}
    for md,cut in CUTS.items():
        seasons={c:(df if c!=S else df.sort_values("date",kind="stable").head(cut)) for c,df in sp1.items() if c<=S}
        second={c:df for c,df in sp2_all.items() if c<S}
        long=ingest.to_long(ingest.add_elo(seasons,second_div=second)); elo_now=ingest.final_elos(seasons,second_div=second)
        played=seasons[S][["home","away","hg","ag"]]; pend=ingest.pending_fixtures(seasons[S])
        for sname,sh in SHRINKS.items():
            m=lm.train_liga(long,shrink=sh); xg=lm.predict_fixtures(m,lm.team_state(long,m,elo_now),pend)
            for sg in SIGMAS:
                out=sim.simulate_season(teams,played,xg,m.rho,N_SIMS,1,strength_sigma=sg)["teams"]
                sc=ev.score_season(out,actual)
                lead=max(out,key=lambda t:out[t]["p_champion"])
                rows.append(dict(season=S,md=md,shrink=sname,sigma=sg,p_lead=out[lead]["p_champion"],lead_won=int(actual[lead]==1),**sc))
        print(f"{S} md{md} listo ({time.time()-t0:.0f}s)",flush=True)
df=pd.DataFrame(rows); df.to_csv(path("results","liga_backtest.csv"),index=False)
print("FIN",flush=True)
