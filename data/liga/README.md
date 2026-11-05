# Datos de La Liga

1. `python scripts/download_liga.py` (en local, con red) baja `SP1_<aabb>.csv` de
   football-data.co.uk a `data/liga/raw/`. También sirve copiarlos a mano.
2. `python src/liga/ingest.py` genera:
   - `df_liga.csv` — formato largo (2 filas por partido, ELO propio previo al partido)
   - `pending_<aabb>.csv` — partidos que faltan de la temporada en curso
   - `elo_actual.csv` — ELO actual de cada club

ELO propio (K=20, ventaja de local 65, inicio 1500). Los ascendidos empiezan con el
ELO medio de los descendidos de la temporada anterior. Los nombres de equipo son los
de football-data.
