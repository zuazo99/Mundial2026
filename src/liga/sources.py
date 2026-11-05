"""Fuentes abiertas de La Liga: openfootball/football.json (dominio público) → CSV limpios.

Descarga el JSON de Primera (es.1) y Segunda (es.2) de cada temporada, lo limpia con
reglas explícitas y escribe `SP1_<aabb>.csv` / `SP2_<aabb>.csv` en formato football-data
para que la ingesta (ingest.py) funcione sin cambios. Todo se cuenta en un informe.

Reglas de limpieza (cada descarte queda contado en el informe):
  * jugado = tiene `score.ft`; si `score` es lista vacía o falta, no está jugado;
  * `ft` = dos enteros entre 0 y 12 y, si hay `ht`, `ht <= ft`; si no, se descarta;
  * resultado con fecha posterior a `today`: se descarta;
  * pares duplicados, equipo contra sí mismo, dos partidos del mismo equipo el mismo día: error;
  * temporada en curso: todo equipo del calendario debe tener ≥1 partido jugado (si no, error);
  * alias: nombres antiguos → nombre reciente (la fuente renombró 10 clubes en 2020/21).

Validación cruzada independiente de los nombres: se compara, fecha a fecha, el multiconjunto
de marcadores con footballcsv/cache.footballdata (espejo de football-data.co.uk; llega a 2023/24).

Fuentes: https://github.com/openfootball/football.json y
https://github.com/footballcsv/cache.footballdata (ambas CC0 / dominio público).
"""
import collections
import datetime as dt
import io
import json
import os
import sys
import urllib.error
import urllib.request

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from liga import ingest  # noqa: E402

RAW_BASE = "https://raw.githubusercontent.com"
OPENFOOTBALL_URL = RAW_BASE + "/openfootball/football.json/master/{season}/es.{level}.json"
CACHE_FD_URL = RAW_BASE + "/footballcsv/cache.footballdata/master/{season}/es.1.csv"
MAX_GOALS = 12

# Nombre antiguo → nombre reciente (derivados de las transiciones reales de la fuente).
ALIASES = {
    "Atlético Madrid": "Club Atlético de Madrid",
    "CD Alavés": "Deportivo Alavés",
    "Espanyol Barcelona": "RCD Espanyol de Barcelona",
    "RC Celta": "RC Celta de Vigo",
    "Real Betis": "Real Betis Balompié",
    "Real Madrid": "Real Madrid CF",
    "Real Sociedad": "Real Sociedad de Fútbol",
    "Real Valladolid": "Real Valladolid CF",
    "Rayo Vallecano": "Rayo Vallecano de Madrid",
    "Deportivo La Coruña": "RC Deportivo La Coruña",
    "Racing Santander": "Real Racing Club de Santander",   # Segunda 2025/26 → Primera 2026/27
}


class NotFound(Exception):
    """El recurso no existe en la fuente (HTTP 404)."""


def season_code(season: str) -> str:
    """'2025-26' → '2526'."""
    a, b = season.split("-")
    return a[2:] + b


def canon(name: str) -> str:
    name = name.strip()
    return ALIASES.get(name, name)


# ─── limpieza de una temporada ───────────────────────────────────────────────

def _valid_ft(ft) -> bool:
    return (isinstance(ft, list) and len(ft) == 2 and
            all(isinstance(x, int) and not isinstance(x, bool) and 0 <= x <= MAX_GOALS for x in ft))


def clean_season(raw: dict, today: dt.date):
    """(DataFrame home/away/hg/ag/date de partidos jugados, estadísticas, equipos del calendario)."""
    matches = raw["matches"]
    pairs = collections.Counter()
    teams, dates = set(), []
    rows, st = [], {"calendar": len(matches), "played": 0, "unplayed": 0, "dropped_future": 0, "dropped_bad_score": 0}
    for m in matches:
        h, a = canon(m["team1"]), canon(m["team2"])
        if h == a:
            raise ValueError(f"Partido de un equipo contra sí mismo: {h}")
        pairs[(h, a)] += 1
        teams |= {h, a}
        d = dt.date.fromisoformat(m["date"])
        dates.append(d)
        sc = m.get("score")
        ft = sc.get("ft") if isinstance(sc, dict) else None
        if ft is None:
            st["unplayed"] += 1
            continue
        ht = sc.get("ht")
        if not _valid_ft(ft) or (ht is not None and _valid_ft(ht) and any(x > y for x, y in zip(ht, ft))) \
                or (ht is not None and not _valid_ft(ht)):
            st["dropped_bad_score"] += 1
            continue
        if d > today:
            st["dropped_future"] += 1
            continue
        rows.append((pd.Timestamp(d), h, a, ft[0], ft[1]))
    dup = [p for p, c in pairs.items() if c > 1]
    if dup:
        raise ValueError(f"Pares duplicados en el calendario (p. ej. {dup[0]}); {len(dup)} en total.")
    df = pd.DataFrame(rows, columns=["date", "home", "away", "hg", "ag"])
    clash = pd.concat([df[["date", "home"]].rename(columns={"home": "t"}),
                       df[["date", "away"]].rename(columns={"away": "t"})])
    same_day = clash[clash.duplicated(["date", "t"], keep=False)]
    if len(same_day):
        raise ValueError(f"{same_day.iloc[0]['t']} juega dos partidos en la misma fecha "
                         f"({same_day.iloc[0]['date'].date()}).")
    st["played"] = len(df)
    st["finished"] = bool(max(dates) < today)
    if not st["finished"]:
        never = sorted(teams - set(df["home"]) - set(df["away"]))
        if never:
            raise ValueError(f"Temporada en curso: sin partidos jugados {never}; "
                             "pending_fixtures no los vería.")
    return df.sort_values("date", kind="stable").reset_index(drop=True), st, teams


def to_footballdata_frame(df: pd.DataFrame, div: str) -> pd.DataFrame:
    return pd.DataFrame({"Div": div, "Date": df["date"].dt.strftime("%d/%m/%Y"),
                         "HomeTeam": df["home"], "AwayTeam": df["away"],
                         "FTHG": df["hg"], "FTAG": df["ag"]})


# ─── informes ────────────────────────────────────────────────────────────────

def _teams(df):
    return set(df["home"]) | set(df["away"])


def transitions(seasons: dict) -> list:
    codes = list(seasons)
    return [{"from": a, "to": b, "out": sorted(_teams(seasons[a]) - _teams(seasons[b])),
             "in": sorted(_teams(seasons[b]) - _teams(seasons[a]))} for a, b in zip(codes, codes[1:])]


def promoted_coverage(sp1: dict, sp2: dict) -> dict:
    """Por temporada: ascendidos y cuántos aparecen con el mismo nombre en la Segunda anterior."""
    out, codes = {}, list(sp1)
    for a, b in zip(codes, codes[1:]):
        promoted = sorted(_teams(sp1[b]) - _teams(sp1[a]))
        avail = a in sp2
        out[b] = {"promoted": promoted,
                  "found_in_sp2": sorted(set(promoted) & _teams(sp2[a])) if avail else [],
                  "sp2_available": avail}
    return out


def crosscheck(df: pd.DataFrame, cache_csv: str) -> dict:
    """Acuerdo de marcadores por fecha con footballcsv/cache.footballdata (sin usar nombres)."""
    other = pd.read_csv(io.StringIO(cache_csv))
    theirs = collections.defaultdict(collections.Counter)
    for d, ft in zip(pd.to_datetime(other["Date"], format="%a %b %d %Y"), other["FT"]):
        if isinstance(ft, str) and "-" in ft:
            h, a = ft.split("-")
            theirs[d.date()][(int(h), int(a))] += 1
    ours = collections.defaultdict(collections.Counter)
    for d, h, a in zip(df["date"], df["hg"], df["ag"]):
        ours[d.date()][(int(h), int(a))] += 1
    matched = sum(sum((ours[d] & theirs[d]).values()) for d in set(ours) | set(theirs))
    n_ours, n_theirs = sum(sum(c.values()) for c in ours.values()), sum(sum(c.values()) for c in theirs.values())
    return {"available": True, "agreement": round(matched / max(n_ours, n_theirs, 1), 4), "n": int(n_ours),
            "n_other": int(n_theirs),
            "mismatched_dates": sum(1 for d in set(ours) | set(theirs) if ours[d] != theirs[d])}


# ─── descarga y construcción ─────────────────────────────────────────────────

def http_fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "mundial2026-liga-sources"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise NotFound(url) from e
        raise


def _try(fetch, url):
    try:
        return fetch(url)
    except NotFound:
        return None


def build(seasons: list, today: dt.date, fetch, raw_dir: str, cache_dir: str, report_path: str) -> dict:
    """Descarga, limpia y escribe SP1/SP2 por temporada + informe. Primera división obligatoria."""
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(os.path.join(cache_dir, "openfootball"), exist_ok=True)
    os.makedirs(os.path.join(cache_dir, "cache.footballdata"), exist_ok=True)
    sp1, sp2, per_season = {}, {}, {}
    for season in seasons:
        code = season_code(season)
        entry = {}
        for level, div, store in ((1, "SP1", sp1), (2, "SP2", sp2)):
            url = OPENFOOTBALL_URL.format(season=season, level=level)
            body = fetch(url) if level == 1 else _try(fetch, url)
            if body is None:
                entry[div] = {"available": False}
                continue
            with open(os.path.join(cache_dir, "openfootball", f"{season}_es.{level}.json"), "wb") as f:
                f.write(body)
            df, st, _ = clean_season(json.loads(body.decode("utf-8")), today)
            store[code] = df
            to_footballdata_frame(df, div).to_csv(os.path.join(raw_dir, f"{div}_{code}.csv"), index=False)
            entry[div] = dict(st, available=True)
        csv = _try(fetch, CACHE_FD_URL.format(season=season))
        if csv is None:
            entry["crosscheck"] = {"available": False}
        else:
            text = csv.decode("utf-8")
            with open(os.path.join(cache_dir, "cache.footballdata", f"{season}_es.1.csv"), "w", encoding="utf-8") as f:
                f.write(text)
            entry["crosscheck"] = crosscheck(sp1[code], text)
        per_season[code] = entry
    ingest.check_transitions(sp1)
    last = list(sp1)[-1]
    report = {"generated_for": today.isoformat(), "data_as_of": sp1[last]["date"].max().date().isoformat(),
              "seasons": per_season, "transitions": transitions(sp1),
              "promoted_coverage": promoted_coverage(sp1, sp2)}
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return report
