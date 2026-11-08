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
import csv
import datetime as dt
import difflib
import hashlib
import io
import json
import os
import re
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


# ═══ Histórico desde el espejo datasets/football-datasets (1993/94 →) ═════════════════════════
# Fuente: https://github.com/datasets/football-datasets (PDDL), espejo diario de football-data.co.uk.
# Se usa solo como HISTÓRICO: no tiene la temporada en curso y contiene filas de relleno (ver quarantine).

MIRROR_REPO = "datasets/football-datasets"
MIRROR_URL = RAW_BASE + "/" + MIRROR_REPO + "/{ref}/datasets/la-liga/season-{code}.csv"
_MIRROR_REQUIRED = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]

# Nombre del espejo (esquema football-data) → nombre canónico (esquema openfootball).
# Derivados por coincidencia (fecha, marcador) con openfootball 2016/17-2025/26 y verificados en cada
# ejecución con check_mirror_aliases; los dos últimos son manuales (clubes sin solape que lo derive).
MIRROR_ALIASES = {
    'Alaves': 'Deportivo Alavés', 'Almeria': 'UD Almería', 'Ath Bilbao': 'Athletic Club',
    'Ath Madrid': 'Club Atlético de Madrid', 'Barcelona': 'FC Barcelona', 'Betis': 'Real Betis Balompié',
    'Cadiz': 'Cádiz CF', 'Celta': 'RC Celta de Vigo', 'Eibar': 'SD Eibar', 'Elche': 'Elche CF',
    'Espanol': 'RCD Espanyol de Barcelona', 'Getafe': 'Getafe CF', 'Girona': 'Girona FC', 'Granada': 'Granada CF',
    'Huesca': 'SD Huesca', 'La Coruna': 'RC Deportivo La Coruña', 'Las Palmas': 'UD Las Palmas',
    'Leganes': 'CD Leganés', 'Levante': 'Levante UD', 'Malaga': 'Málaga CF', 'Mallorca': 'RCD Mallorca',
    'Osasuna': 'CA Osasuna', 'Oviedo': 'Real Oviedo', 'Real Madrid': 'Real Madrid CF', 'Sevilla': 'Sevilla FC',
    'Sociedad': 'Real Sociedad de Fútbol', 'Sp Gijon': 'Sporting Gijón', 'Valencia': 'Valencia CF',
    'Valladolid': 'Real Valladolid CF', 'Vallecano': 'Rayo Vallecano de Madrid', 'Villarreal': 'Villarreal CF',
    # manuales (sin solape con openfootball 2016+ que permita derivarlos):
    'Santander': 'Real Racing Club de Santander',    # único club "Santander" de La Liga (2008/09-2011/12 en el espejo)
    'Villareal': 'Villarreal CF',                    # errata del espejo en las temporadas anteriores a 2000
}

# Pares de nombres parecidos ya revisados a mano: son clubes distintos.
KNOWN_DISTINCT = {frozenset(p) for p in [("Lerida", "Merida"), ("Real Madrid CF", "Real Valladolid CF")]}


class SourceChanged(Exception):
    """Un fichero descargado no coincide con la huella (SHA-256) registrada."""


_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def mirror_url(code: str, ref: str = "master") -> str:
    return MIRROR_URL.format(ref=ref, code=code)


def parse_mirror(text: str) -> dict:
    """CSV del espejo → dict con la forma de openfootball (para reutilizar clean_season). Fechas ISO estrictas."""
    reader = csv.DictReader(io.StringIO(text.lstrip("\ufeff")))
    missing = [c for c in _MIRROR_REQUIRED if c not in (reader.fieldnames or [])]
    if missing:
        raise ValueError(f"Espejo: faltan columnas {missing}")
    matches = []
    for r in reader:
        d = (r["Date"] or "").strip()
        if not _ISO.match(d):
            raise ValueError(f"Espejo: fecha no ISO {d!r}")
        dt.date.fromisoformat(d)                                   # valida mes/día
        h, a = (r["FTHG"] or "").strip(), (r["FTAG"] or "").strip()
        m = {"round": "", "date": d, "team1": r["HomeTeam"].strip(), "team2": r["AwayTeam"].strip()}
        if h != "" and a != "":
            m["score"] = {"ft": [int(h) if h.lstrip("-").isdigit() else h, int(a) if a.lstrip("-").isdigit() else a]}
        matches.append(m)
    return {"matches": matches}


def apply_mirror_aliases(raw: dict) -> dict:
    out = []
    for m in raw["matches"]:
        m = dict(m)
        m["team1"], m["team2"] = MIRROR_ALIASES.get(m["team1"], m["team1"]), MIRROR_ALIASES.get(m["team2"], m["team2"])
        out.append(m)
    return {"matches": out}


def derive_aliases(mirror: dict, ref: dict) -> dict:
    """Alias espejo → referencia por coincidencia de (fecha, marcador) únicos en ambas fuentes. Conflicto → error."""
    votes = collections.defaultdict(collections.Counter)
    for code in sorted(set(mirror) & set(ref)):
        m_count = collections.Counter((r.date, r.hg, r.ag) for r in mirror[code].itertuples())
        idx = collections.defaultdict(list)
        for r in ref[code].itertuples():
            idx[(r.date, r.hg, r.ag)].append((r.home, r.away))
        for r in mirror[code].itertuples():
            key = (r.date, r.hg, r.ag)
            if m_count[key] == 1 and len(idx.get(key, [])) == 1:
                votes[r.home][idx[key][0][0]] += 1
                votes[r.away][idx[key][0][1]] += 1
    out = {}
    for name, c in votes.items():
        (best, n), *rest = c.most_common()
        if rest and rest[0][1] >= max(2, 0.2 * n):
            raise ValueError(f"Alias en conflicto para {name!r}: {dict(c)}")
        if best != name:
            out[name] = best
    return out


def check_mirror_aliases(mirror: dict, ref: dict) -> dict:
    """Comprueba que el mapa estático MIRROR_ALIASES coincide con lo que dicen los datos."""
    derived = derive_aliases(mirror, ref)
    for name, target in derived.items():
        if MIRROR_ALIASES.get(name, name) != target:
            raise ValueError(f"Alias {name!r}: los datos dicen {target!r} pero MIRROR_ALIASES dice "
                             f"{MIRROR_ALIASES.get(name, name)!r}")
    return derived


def similar_names(names, threshold: float = 0.75) -> list:
    """Pares de nombres parecidos aún no revisados: posibles renombres que el recuento de entradas no detecta."""
    ns = sorted(set(names))
    return [(a, b) for i, a in enumerate(ns) for b in ns[i + 1:]
            if frozenset((a, b)) not in KNOWN_DISTINCT
            and difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio() >= threshold]


def uncorroborated(mirror_df: pd.DataFrame, ref_df: pd.DataFrame) -> pd.DataFrame:
    """Filas del espejo que la fuente de referencia no respalda (misma fecha y marcador, sin usar nombres)."""
    avail = collections.defaultdict(collections.Counter)
    for r in ref_df.itertuples():
        avail[r.date][(r.hg, r.ag)] += 1
    extra = []
    for r in mirror_df.itertuples(index=False):
        if avail[r.date][(r.hg, r.ag)] > 0:
            avail[r.date][(r.hg, r.ag)] -= 1
        else:
            extra.append(r)
    return pd.DataFrame(extra, columns=["date", "home", "away", "hg", "ag"]) if extra else mirror_df.iloc[0:0].copy()


def quarantine(extra: pd.DataFrame, min_rows: int = 3, zero_share: float = 0.8):
    """Separa (conservadas, en cuarentena) las filas de fuente única.

    Si hay al menos `min_rows` y ≥ `zero_share` son 0-0 se ponen todas en cuarentena: en 2025/26 el espejo
    trae 15 resultados que openfootball no tiene y los 15 son 0-0 (la frecuencia real de 0-0 es ≈ 5 %).
    """
    if len(extra) >= min_rows and ((extra["hg"] == 0) & (extra["ag"] == 0)).mean() >= zero_share:
        return extra.iloc[0:0].copy(), extra.copy()
    return extra.copy(), extra.iloc[0:0].copy()


def merge_season(ref_df: pd.DataFrame, mirror_df: pd.DataFrame):
    """Referencia + filas de fuente única que superan la cuarentena. Devuelve (DataFrame, estadísticas)."""
    kept, quarantined = quarantine(uncorroborated(mirror_df, ref_df))
    merged = pd.concat([ref_df, kept], ignore_index=True).sort_values("date", kind="stable").reset_index(drop=True)
    return merged, {"reference": int(len(ref_df)), "added_from_mirror": int(len(kept)), "quarantined": int(len(quarantined))}


def verify_pinned(code: str, body: bytes, expected: dict, accept_changes: bool = False) -> str:
    """SHA-256 del fichero; si había huella registrada y no coincide, falla salvo accept_changes."""
    sha = hashlib.sha256(body).hexdigest()
    exp = expected.get(code)
    if exp and exp != sha and not accept_changes:
        raise SourceChanged(f"{code}: la huella cambió (esperada {exp[:12]}…, obtenida {sha[:12]}…). "
                            "Revisa el cambio y repite con --accept-changes si es legítimo.")
    return sha


def _frame(raw: dict) -> pd.DataFrame:
    """Partidos con resultado de un dict estilo openfootball, SIN aplicar alias (para derivarlos de los datos)."""
    rows = [(pd.Timestamp(m["date"]), m["team1"], m["team2"], m["score"]["ft"][0], m["score"]["ft"][1])
            for m in raw["matches"] if m.get("score", {}).get("ft") and all(isinstance(x, int) for x in m["score"]["ft"])]
    return pd.DataFrame(rows, columns=["date", "home", "away", "hg", "ag"])


def build_history(mirror_bodies: dict, ref_dfs: dict, today: dt.date, expected_hashes: dict = None,
                  accept_changes: bool = False):
    """Primera división 1993/94 → hoy: espejo para el histórico + openfootball (limpio) donde existe.

    Devuelve ({código: DataFrame}, informe). Falla ruidosamente si: cambia la huella de un fichero, un alias
    estático contradice los datos, hay pares de nombres parecidos sin revisar o una transición de equipos
    es sospechosa.
    """
    expected = expected_hashes or {}
    shas, mir, mir_stats, frames = {}, {}, {}, {}
    for code, body in mirror_bodies.items():
        shas[code] = verify_pinned(code, body, expected, accept_changes)
        raw = parse_mirror(body.decode("utf-8-sig"))
        frames[code] = _frame(raw)
        mir[code], mir_stats[code], _ = clean_season(apply_mirror_aliases(raw), today)
    derived = check_mirror_aliases(frames, ref_dfs)

    seasons, per_season, quarantined = {}, {}, []
    for code in sorted(set(mir) | set(ref_dfs), key=ingest.season_key):
        if code in mir and code in ref_dfs:
            seasons[code], st = merge_season(ref_dfs[code], mir[code])
            _, q = quarantine(uncorroborated(mir[code], ref_dfs[code]))
            quarantined += [{"season": code, "date": str(r.date.date()), "home": r.home, "away": r.away,
                             "hg": int(r.hg), "ag": int(r.ag)} for r in q.itertuples()]
            per_season[code] = {"source": "merged", **st}
        elif code in mir:
            seasons[code] = mir[code]
            per_season[code] = {"source": "mirror", "matches": int(len(mir[code])), **{k: mir_stats[code][k] for k in ("unplayed", "dropped_bad_score")}}
        else:
            seasons[code] = ref_dfs[code]
            per_season[code] = {"source": "openfootball", "matches": int(len(ref_dfs[code]))}
    names = {t for df in seasons.values() for t in set(df["home"]) | set(df["away"])}
    flagged = similar_names(names)
    if flagged:
        raise ValueError(f"Pares de nombres parecidos sin revisar (¿renombre o errata?): {flagged}")
    ingest.check_transitions(seasons)
    report = {"mirror_sha256": shas, "seasons": per_season, "quarantined": quarantined,
              "aliases_verified_against_data": len(derived),
              "aliases_manual_unverified": sorted(k for k in MIRROR_ALIASES if k not in derived and any(
                  k in set(f["home"]) | set(f["away"]) for f in frames.values())),
              "n_clubs": len(names)}
    return seasons, report
