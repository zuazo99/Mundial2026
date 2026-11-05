"""Ingesta de La Liga (football-data.co.uk, ficheros SP1_<aamm>.csv).

Pasos:
  1. read_season / load_seasons  → partidos jugados por temporada.
  2. add_elo                     → ELO propio previo a cada partido (sin clubelo).
  3. to_long                     → formato largo (2 filas por partido) que consume
                                   el pipeline de xg_preds (engineer_features).
  4. pending_fixtures            → partidos que faltan = pares (local, visitante)
                                   del doble round-robin menos los jugados.

Los nombres de equipo son los de football-data (canónicos). Uso:
    python src/liga/ingest.py            # lee data/liga/raw/, escribe data/liga/
"""
import glob
import itertools
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import path  # noqa: E402

TOURNAMENT = "La Liga"
REQUIRED_COLS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]

# ELO propio (estilo clubelo, sin margen de goles).
ELO_INIT = 1500.0
ELO_K = 20.0
ELO_HOME_ADV = 65.0


# ─── 1. parser ───────────────────────────────────────────────────────────────

def _read_text_csv(file_path: str) -> pd.DataFrame:
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(file_path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"No se pudo decodificar {file_path}")


def read_season(file_path: str) -> pd.DataFrame:
    """Partidos jugados de una temporada: date, home, away, hg, ag."""
    raw = _read_text_csv(file_path)
    missing = [c for c in REQUIRED_COLS if c not in raw.columns]
    if missing:
        raise ValueError(f"{os.path.basename(file_path)}: faltan columnas {missing}")
    raw = raw.dropna(subset=REQUIRED_COLS)
    df = pd.DataFrame({
        "date": pd.to_datetime(raw["Date"], dayfirst=True, format="mixed"),
        "home": raw["HomeTeam"].astype(str).str.strip(),
        "away": raw["AwayTeam"].astype(str).str.strip(),
        "hg": raw["FTHG"].astype(int),
        "ag": raw["FTAG"].astype(int),
    })
    df = df.drop_duplicates(subset=["date", "home", "away"])
    return df.sort_values("date", kind="stable").reset_index(drop=True)


def load_seasons(raw_dir: str) -> dict:
    """{código_temporada: DataFrame} ordenado cronológicamente (SP1_2122.csv → '2122')."""
    seasons = {}
    for f in sorted(glob.glob(os.path.join(raw_dir, "SP1_*.csv"))):
        code = os.path.splitext(os.path.basename(f))[0].split("_", 1)[1]
        seasons[code] = read_season(f)
    return dict(sorted(seasons.items()))


# ─── 2. ELO propio ───────────────────────────────────────────────────────────

def elo_delta(elo_home: float, elo_away: float, hg: int, ag: int) -> float:
    """Cambio de ELO del local (el visitante recibe el opuesto → suma cero)."""
    expected = 1.0 / (1.0 + 10 ** (-(elo_home + ELO_HOME_ADV - elo_away) / 400.0))
    result = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
    return ELO_K * (result - expected)


def _run_elo(seasons: dict):
    """Devuelve (partidos con ELO previo, ELO final por equipo).

    Entre temporadas se arrastra el ELO. Los ascendidos (sin historia en la
    temporada anterior) empiezan con el ELO medio de los descendidos.
    """
    ratings: dict = {}
    prev_teams: set = set()
    frames = []
    for code, df in seasons.items():
        teams = set(df["home"]) | set(df["away"])
        relegated = [ratings[t] for t in prev_teams - teams if t in ratings]
        promoted_elo = sum(relegated) / len(relegated) if relegated else ELO_INIT
        ratings = {t: (ratings[t] if t in prev_teams else (promoted_elo if prev_teams else ELO_INIT))
                   for t in teams}
        rows = []
        for r in df.itertuples(index=False):
            eh, ea = ratings[r.home], ratings[r.away]
            rows.append((code, r.date, r.home, r.away, r.hg, r.ag, eh, ea))
            d = elo_delta(eh, ea, r.hg, r.ag)
            ratings[r.home] = eh + d
            ratings[r.away] = ea - d
        frames.append(pd.DataFrame(rows, columns=["season", "date", "home", "away", "hg", "ag",
                                                  "elo_home", "elo_away"]))
        prev_teams = teams
    matches = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["season", "date", "home", "away", "hg", "ag", "elo_home", "elo_away"])
    return matches, ratings


def add_elo(seasons: dict) -> pd.DataFrame:
    return _run_elo(seasons)[0]


def final_elos(seasons: dict) -> dict:
    return _run_elo(seasons)[1]


# ─── 3. formato largo ────────────────────────────────────────────────────────

def to_long(matches: pd.DataFrame) -> pd.DataFrame:
    """Dos filas por partido con el esquema que consume xg_preds."""
    home = pd.DataFrame({
        "date": matches["date"], "team": matches["home"], "opponent": matches["away"],
        "goals": matches["hg"], "goals_conceded": matches["ag"], "is_home": 1,
        "elo": matches["elo_home"], "opponent_elo": matches["elo_away"],
        "season": matches["season"],
    })
    away = pd.DataFrame({
        "date": matches["date"], "team": matches["away"], "opponent": matches["home"],
        "goals": matches["ag"], "goals_conceded": matches["hg"], "is_home": 0,
        "elo": matches["elo_away"], "opponent_elo": matches["elo_home"],
        "season": matches["season"],
    })
    long = pd.concat([home, away], ignore_index=True)
    long["tournament"] = TOURNAMENT
    return long.sort_values(["date", "team"], kind="stable").reset_index(drop=True)


# ─── 4. calendario pendiente ─────────────────────────────────────────────────

def pending_fixtures(season_df: pd.DataFrame) -> pd.DataFrame:
    """Pares (local, visitante) del doble round-robin que aún no se han jugado.

    Los equipos se sacan del propio fichero de la temporada; los aplazados
    quedan como pendientes sin tratamiento especial.
    """
    teams = sorted(set(season_df["home"]) | set(season_df["away"]))
    played = set(zip(season_df["home"], season_df["away"]))
    rows = [(h, a) for h, a in itertools.permutations(teams, 2) if (h, a) not in played]
    return pd.DataFrame(rows, columns=["home", "away"])


# ─── CLI ─────────────────────────────────────────────────────────────────────

def build_dataset(raw_dir: str, out_dir: str) -> None:
    seasons = load_seasons(raw_dir)
    if not seasons:
        raise SystemExit(f"No hay ficheros SP1_*.csv en {raw_dir}. "
                         "Descárgalos con scripts/download_liga.py o cópialos a mano.")
    matches = add_elo(seasons)
    long = to_long(matches)
    os.makedirs(out_dir, exist_ok=True)
    long.to_csv(os.path.join(out_dir, "df_liga.csv"), index=False)
    last = list(seasons)[-1]
    pending = pending_fixtures(seasons[last])
    pending.to_csv(os.path.join(out_dir, f"pending_{last}.csv"), index=False)
    elo = pd.Series(final_elos(seasons)).sort_values(ascending=False).round(1)
    elo.rename("elo").rename_axis("team").to_csv(os.path.join(out_dir, "elo_actual.csv"))
    print(f"✅ {len(long)//2} partidos en {len(seasons)} temporadas → {out_dir}/df_liga.csv")
    print(f"✅ {len(pending)} partidos pendientes de {last} → pending_{last}.csv")


if __name__ == "__main__":
    build_dataset(path("data", "liga", "raw"), path("data", "liga"))
