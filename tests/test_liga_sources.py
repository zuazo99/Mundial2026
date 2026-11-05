"""Fuentes abiertas de La Liga: adaptador openfootball, limpieza, alias, informe y validación cruzada."""
import datetime as dt
import itertools
import json
import os

import pandas as pd
import pytest

from liga import ingest, sources
from liga_synth import _round_robin_rounds

TODAY = dt.date(2025, 12, 1)
TEAMS4 = ["Alpha", "Beta", "Gamma", "Delta"]


def match(date, t1, t2, ft=None, ht=None, as_list=False):
    m = {"round": "Matchday 1", "date": date, "team1": t1, "team2": t2}
    if ft is not None:
        m["score"] = {"ht": ht, "ft": ft} if ht else {"ft": ft}
    elif as_list:
        m["score"] = []
    return m


def league(start, teams=TEAMS4, played_until=None, missing=()):
    """Doble vuelta semanal válida (un partido por equipo y jornada); sin resultado después de
    played_until y en los índices de `missing`."""
    first = _round_robin_rounds(teams)
    rounds = first + [[(b, a) for a, b in r] for r in first]
    out, d0, i = [], dt.date.fromisoformat(start), 0
    for ri, r in enumerate(rounds):
        d = d0 + dt.timedelta(days=7 * ri)
        for a, b in r:
            played = (played_until is None or d <= dt.date.fromisoformat(played_until)) and i not in missing
            out.append(match(d.isoformat(), a, b, ft=[i % 3, (i * 2) % 3] if played else None))
            i += 1
    return {"name": "Test", "matches": out}


def csv_for(raw):
    """Réplica del formato footballcsv/cache.footballdata con nombres distintos (cortos)."""
    short = {"Alpha": "Alp", "Beta": "Bet", "Gamma": "Gam", "Delta": "Del"}
    rows = ["Date,Team 1,FT,HT,Team 2"]
    for m in raw["matches"]:
        if m.get("score") and m["score"].get("ft"):
            d = dt.date.fromisoformat(m["date"])
            rows.append(f"{d.strftime('%a %b %d %Y')},{short[m['team1']]},{m['score']['ft'][0]}-{m['score']['ft'][1]},0-0,{short[m['team2']]}")
    return "\n".join(rows)


# ─── limpieza ──────────────────────────────────────────────────────────────

def test_season_code():
    assert sources.season_code("2025-26") == "2526"
    assert sources.season_code("1999-00") == "9900"


def test_score_as_dict_list_or_missing_counts_as_played_only_when_ft_present():
    raw = {"matches": [match("2025-09-01", "Alpha", "Beta", ft=[1, 0]),
                       match("2025-09-01", "Gamma", "Delta", as_list=True),   # score: []
                       match("2025-09-08", "Beta", "Alpha"),                   # sin score
                       match("2025-09-08", "Delta", "Gamma", ft=[2, 2])]}
    df, st, _ = sources.clean_season(raw, TODAY)
    assert len(df) == 2 and st["played"] == 2 and st["unplayed"] == 2 and st["calendar"] == 4


def test_result_dated_in_the_future_is_dropped_and_counted():
    raw = {"matches": [match("2025-09-01", "Alpha", "Beta", ft=[1, 0]),
                       match("2025-09-08", "Gamma", "Delta", ft=[0, 0]),
                       match("2026-03-01", "Beta", "Alpha", ft=[2, 0])]}   # marcador en el futuro: imposible
    df, st, _ = sources.clean_season(raw, TODAY)
    assert len(df) == 2 and st["dropped_future"] == 1


@pytest.mark.parametrize("ft,ht", [([13, 0], None), ([-1, 2], None), ([1, "x"], None), ([1], None),
                                   ([1, 0], [2, 0])])      # ht > ft
def test_invalid_scores_are_dropped_and_counted(ft, ht):
    raw = {"matches": [match("2025-09-01", "Alpha", "Beta", ft=[1, 0]),
                       match("2025-09-08", "Gamma", "Delta", ft=ft, ht=ht)]}
    df, st, _ = sources.clean_season(raw, TODAY)
    assert len(df) == 1 and st["dropped_bad_score"] == 1


def test_aliases_map_old_names_to_canonical_ones():
    raw = {"matches": [match("2025-09-01", "Real Madrid", "Atlético Madrid", ft=[2, 1])]}
    df, _, teams = sources.clean_season(raw, TODAY)
    assert df.iloc[0]["home"] == "Real Madrid CF" and df.iloc[0]["away"] == "Club Atlético de Madrid"
    assert "Real Madrid CF" in teams


@pytest.mark.parametrize("matches,msg", [
    ([match("2025-09-01", "Alpha", "Beta", ft=[1, 0]), match("2025-09-08", "Alpha", "Beta", ft=[0, 0])], "duplicad"),
    ([match("2025-09-01", "Alpha", "Alpha", ft=[1, 0])], "contra sí mismo"),
    ([match("2025-09-01", "Alpha", "Beta", ft=[1, 0]), match("2025-09-01", "Alpha", "Gamma", ft=[0, 0])], "misma fecha"),
])
def test_structural_problems_raise(matches, msg):
    with pytest.raises(ValueError, match=msg):
        sources.clean_season({"matches": matches}, TODAY)


def test_finished_season_reports_missing_results_and_current_season_reports_pending():
    old = league("2024-08-15", missing={0, 5})
    _, st, _ = sources.clean_season(old, TODAY)
    assert st["finished"] is True and st["unplayed"] == 2 and st["played"] == 10
    cur = league("2025-11-01", played_until="2025-11-15")
    _, st2, _ = sources.clean_season(cur, TODAY)
    assert st2["finished"] is False and st2["played"] + st2["unplayed"] == 12


def test_current_season_team_without_any_played_match_is_an_error():
    raw = league("2025-11-01", played_until="2025-11-08")
    raw["matches"] = [m for m in raw["matches"] if not (m.get("score") and "Delta" in (m["team1"], m["team2"]))]
    with pytest.raises(ValueError, match="Delta"):
        sources.clean_season(raw, TODAY)


# ─── transiciones y ascensos ───────────────────────────────────────────────

def _season_df(teams):
    return pd.DataFrame([(pd.Timestamp("2024-01-01"), a, b, 1, 0) for a, b in itertools.permutations(teams, 2)],
                        columns=["date", "home", "away", "hg", "ag"])


def test_check_transitions_accepts_normal_promotions_and_rejects_mass_renames():
    a = _season_df(["A", "B", "C", "D", "E", "F"])
    b = _season_df(["A", "B", "C", "D", "X", "Y"])                    # 2 cambios: normal
    ingest.check_transitions({"2122": a, "2223": b}, max_changes=3)
    c = _season_df(["Q", "R", "S", "T", "U", "V"])                     # todo distinto: renombre masivo
    with pytest.raises(ValueError, match="renombre|alias") as e:
        ingest.check_transitions({"2122": a, "2223": c}, max_changes=3)
    assert "Q" in str(e.value) and "A" in str(e.value)


def test_build_dataset_rejects_mass_rename_between_seasons(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for code, teams in (("2122", list("ABCDEF")), ("2223", list("QRSTUV"))):
        pd.DataFrame({"Div": "SP1", "Date": "01/09/2022", "HomeTeam": [a for a, b in itertools.permutations(teams, 2)],
                      "AwayTeam": [b for a, b in itertools.permutations(teams, 2)], "FTHG": 1, "FTAG": 0}
                     ).to_csv(raw / f"SP1_{code}.csv", index=False)
    with pytest.raises(ValueError, match="renombre|alias"):
        ingest.build_dataset(str(raw), str(tmp_path / "out"))


def test_promoted_coverage_measures_name_matches_with_previous_segunda():
    sp1 = {"2425": _season_df(["A", "B", "C", "D"]), "2526": _season_df(["A", "B", "X", "Y"])}
    sp2 = {"2425": _season_df(["X", "Z", "W", "V"])}                   # X coincide; Y no aparece
    cov = sources.promoted_coverage(sp1, sp2)
    assert cov["2526"] == {"promoted": ["X", "Y"], "found_in_sp2": ["X"], "sp2_available": True}
    assert sources.promoted_coverage(sp1, {})["2526"]["sp2_available"] is False


# ─── validación cruzada independiente de nombres ───────────────────────────

def test_crosscheck_agreement_is_name_independent_and_detects_wrong_scores():
    raw = league("2024-08-15")
    df, _, _ = sources.clean_season(raw, TODAY)
    ok = sources.crosscheck(df, csv_for(raw))
    assert ok["agreement"] == 1.0 and ok["n"] == 12
    bad = json.loads(json.dumps(raw))
    bad["matches"][0]["score"]["ft"] = [9, 9]
    res = sources.crosscheck(df, csv_for(bad))
    assert res["agreement"] < 1.0 and res["mismatched_dates"] >= 1


# ─── descarga + informe + CSV de salida ────────────────────────────────────

class FakeNet:
    def __init__(self, files):
        self.files, self.calls = files, []

    def __call__(self, url):
        self.calls.append(url)
        for key, body in self.files.items():
            if url.endswith(key):
                return body.encode() if isinstance(body, str) else json.dumps(body).encode()
        raise sources.NotFound(url)


def test_build_writes_clean_footballdata_csvs_cache_and_report(tmp_path):
    old, cur = league("2024-08-15", missing={3}), league("2025-11-01", played_until="2025-11-15")
    sp2_old = league("2024-08-15", teams=["Alpha", "Epsilon", "Zeta", "Eta"])
    net = FakeNet({"2024-25/es.1.json": old, "2025-26/es.1.json": cur, "2024-25/es.2.json": sp2_old,
                   "2024-25/es.1.csv": csv_for(old)})            # Segunda 2025-26 y la caché de 2025-26 no existen (404)
    paths = dict(raw_dir=str(tmp_path / "raw"), cache_dir=str(tmp_path / "cache"), report_path=str(tmp_path / "rep.json"))
    rep = sources.build(["2024-25", "2025-26"], TODAY, net, **paths)

    sp1 = ingest.read_season(str(tmp_path / "raw" / "SP1_2425.csv"))        # lo lee la ingesta tal cual
    assert len(sp1) == 11                                                     # 12 - 1 sin resultado
    assert (tmp_path / "raw" / "SP2_2425.csv").exists() and not (tmp_path / "raw" / "SP2_2526.csv").exists()
    assert (tmp_path / "cache" / "openfootball" / "2024-25_es.1.json").exists()
    saved = json.load(open(paths["report_path"]))
    assert saved == rep
    assert rep["seasons"]["2425"]["SP1"]["unplayed"] == 1 and rep["seasons"]["2425"]["SP1"]["finished"] is True
    assert rep["seasons"]["2526"]["SP2"] == {"available": False}
    assert rep["data_as_of"] == "2025-11-15"
    assert rep["seasons"]["2425"]["crosscheck"]["agreement"] == 1.0
    assert "promoted_coverage" in rep and "transitions" in rep


def test_build_fails_loudly_when_the_first_division_of_a_season_is_missing(tmp_path):
    net = FakeNet({})
    with pytest.raises(sources.NotFound):
        sources.build(["2024-25"], TODAY, net, raw_dir=str(tmp_path / "r"), cache_dir=str(tmp_path / "c"),
                      report_path=str(tmp_path / "rep.json"))


def test_racing_santander_alias_links_segunda_name_to_primera_name():
    raw = {"matches": [match("2025-09-01", "Racing Santander", "Málaga CF", ft=[1, 0])]}
    df, _, _ = sources.clean_season(raw, TODAY)
    assert df.iloc[0]["home"] == "Real Racing Club de Santander"
