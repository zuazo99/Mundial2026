"""Histórico de La Liga desde el espejo datasets/football-datasets: lectura, alias, cuarentena, fusión y versiones fijadas."""
import datetime as dt
import hashlib
import itertools
import json

import pandas as pd
import pytest

from liga import ingest, sources

TODAY = dt.date(2026, 11, 8)
HEADER = "Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HTHG,HTAG,HTR,Referee,HS,AS,HST,AST,HF,AF,HC,AC,HY,AY,HR,AR"


def mirror_csv(rows):
    """rows: (iso_date, home, away, hg, ag) → CSV con el esquema del espejo (hg/ag '' = sin resultado)."""
    lines = [HEADER]
    for d, h, a, g, c in rows:
        lines.append(f"{d},{h},{a},{g},{c},X,0,0,D,Ref,10,8,4,3,12,11,5,4,1,2,0,0")
    return "\n".join(lines) + "\n"


def df_of(rows):
    return pd.DataFrame([(pd.Timestamp(d), h, a, g, c) for d, h, a, g, c in rows], columns=["date", "home", "away", "hg", "ag"])


# ─── lectura del espejo ────────────────────────────────────────────────────

def test_parse_mirror_reads_iso_dates_strictly():
    raw = sources.parse_mirror(mirror_csv([("1993-09-05", "Ath Madrid", "Betis", 2, 1)]))
    assert raw["matches"][0]["date"] == "1993-09-05"           # 5 de septiembre, no 9 de mayo
    assert raw["matches"][0]["score"]["ft"] == [2, 1]
    df, _, _ = sources.clean_season(raw, TODAY)
    assert df.iloc[0]["date"] == pd.Timestamp("1993-09-05")


@pytest.mark.parametrize("bad", ["05/09/1993", "1993-9-5", "Sun Sep 05 1993", ""])
def test_parse_mirror_rejects_non_iso_dates(bad):
    with pytest.raises(ValueError, match="fecha"):
        sources.parse_mirror(mirror_csv([(bad, "A", "B", 1, 0)]))


def test_parse_mirror_treats_empty_score_as_unplayed_and_requires_columns():
    raw = sources.parse_mirror(mirror_csv([("2025-09-01", "A", "B", "", "")]))
    assert "score" not in raw["matches"][0] or not raw["matches"][0]["score"].get("ft")
    with pytest.raises(ValueError, match="columnas"):
        sources.parse_mirror("Date,HomeTeam\n2025-09-01,A\n")


# ─── alias del espejo ──────────────────────────────────────────────────────

def test_static_mirror_aliases_cover_known_splits():
    assert sources.MIRROR_ALIASES["Villareal"] == "Villarreal CF"          # errata del espejo en temporadas antiguas
    assert sources.MIRROR_ALIASES["Santander"] == "Real Racing Club de Santander"
    assert sources.MIRROR_ALIASES["Ath Madrid"] == "Club Atlético de Madrid"
    assert all(v == sources.canon(v) for v in sources.MIRROR_ALIASES.values())    # destinos ya canónicos


def test_apply_mirror_aliases_changes_only_known_names():
    raw = sources.parse_mirror(mirror_csv([("2000-01-01", "Ath Madrid", "Numancia", 1, 0)]))
    mapped = sources.apply_mirror_aliases(raw)
    assert mapped["matches"][0]["team1"] == "Club Atlético de Madrid" and mapped["matches"][0]["team2"] == "Numancia"


def test_derive_aliases_by_unique_date_and_score_and_detect_conflicts():
    mirror = {"2425": df_of([("2024-09-01", "Ath Madrid", "Betis", 3, 1), ("2024-09-08", "Betis", "Ath Madrid", 0, 2)])}
    ref = {"2425": df_of([("2024-09-01", "Club Atlético de Madrid", "Real Betis Balompié", 3, 1),
                          ("2024-09-08", "Real Betis Balompié", "Club Atlético de Madrid", 0, 2)])}
    assert sources.derive_aliases(mirror, ref) == {"Ath Madrid": "Club Atlético de Madrid", "Betis": "Real Betis Balompié"}
    # misma fecha y marcador repetidos → ambiguo: se ignora, no se adivina
    mirror2 = {"2425": df_of([("2024-09-01", "A", "B", 1, 0), ("2024-09-01", "C", "D", 1, 0)])}
    ref2 = {"2425": df_of([("2024-09-01", "X", "Y", 1, 0), ("2024-09-01", "Z", "W", 1, 0)])}
    assert sources.derive_aliases(mirror2, ref2) == {}
    # dos votos opuestos y fuertes para el mismo nombre → error
    mirror3 = {"2425": df_of([(f"2024-09-{d:02d}", "A", f"R{d}", d % 3, 0) for d in range(1, 9)])}
    ref3 = {"2425": df_of([(f"2024-09-{d:02d}", "X" if d <= 4 else "Y", f"R{d}", d % 3, 0) for d in range(1, 9)])}
    with pytest.raises(ValueError, match="conflicto"):
        sources.derive_aliases(mirror3, ref3)


def test_check_mirror_aliases_fails_when_static_map_disagrees_with_data():
    mirror = {"2425": df_of([("2024-09-01", "Ath Madrid", "Betis", 3, 1)])}
    ref = {"2425": df_of([("2024-09-01", "Atlético OTRO", "Real Betis Balompié", 3, 1)])}
    with pytest.raises(ValueError, match="Ath Madrid"):
        sources.check_mirror_aliases(mirror, ref)


# ─── nombres parecidos ─────────────────────────────────────────────────────

def test_similar_names_flags_unreviewed_pairs_only():
    names = ["Villareal", "Villarreal CF", "Lerida", "Merida", "Real Madrid CF", "Real Valladolid CF", "Sevilla FC"]
    flagged = sources.similar_names(names)
    assert ("Villareal", "Villarreal CF") in flagged
    assert ("Lerida", "Merida") not in flagged and ("Real Madrid CF", "Real Valladolid CF") not in flagged   # revisados
    assert not any("Sevilla FC" in p for p in flagged)


# ─── cuarentena de filas de fuente única ───────────────────────────────────

def test_extras_are_rows_not_corroborated_by_the_reference_source():
    ref = df_of([("2025-09-01", "A", "B", 1, 0)])
    mir = df_of([("2025-09-01", "A", "B", 1, 0), ("2025-09-02", "C", "D", 2, 2)])
    extra = sources.uncorroborated(mir, ref)
    assert list(zip(extra["home"], extra["away"])) == [("C", "D")]


def test_all_zero_zero_cluster_of_extras_is_quarantined_but_plausible_extras_are_kept():
    zeros = df_of([(f"2026-01-{d:02d}", f"H{d}", f"A{d}", 0, 0) for d in range(1, 16)])
    kept, quarantined = sources.quarantine(zeros)
    assert kept.empty and len(quarantined) == 15
    mixed = df_of([("2025-05-23", "A", "B", 1, 1), ("2025-05-24", "C", "D", 2, 0), ("2025-05-24", "E", "F", 3, 0),
                   ("2025-05-25", "G", "H", 0, 0), ("2025-05-25", "I", "J", 0, 4)])
    kept2, q2 = sources.quarantine(mixed)
    assert len(kept2) == 5 and q2.empty
    few = df_of([("2025-05-23", "A", "B", 0, 0), ("2025-05-24", "C", "D", 0, 0)])        # < 3 filas: no hay base para sospechar
    assert len(sources.quarantine(few)[0]) == 2


# ─── fusión ────────────────────────────────────────────────────────────────

def test_merge_season_adds_only_kept_extras_without_duplicates():
    ref = df_of([("2025-09-01", "A", "B", 1, 0)])
    mir = df_of([("2025-09-01", "A", "B", 1, 0), ("2025-09-08", "C", "D", 2, 1)])
    merged, st = sources.merge_season(ref, mir)
    assert len(merged) == 2 and st == {"reference": 1, "added_from_mirror": 1, "quarantined": 0}
    assert merged["date"].is_monotonic_increasing
    again, _ = sources.merge_season(merged, mir)               # idempotente
    assert len(again) == 2


# ─── versiones fijadas ─────────────────────────────────────────────────────

def test_verify_pinned_detects_changed_files_unless_accepted():
    body = b"Date,HomeTeam\n"
    good = hashlib.sha256(body).hexdigest()
    assert sources.verify_pinned("9394", body, {"9394": good}) == good
    with pytest.raises(sources.SourceChanged, match="9394"):
        sources.verify_pinned("9394", body + b"x", {"9394": good})
    assert sources.verify_pinned("9394", body + b"x", {"9394": good}, accept_changes=True) != good
    assert sources.verify_pinned("9495", body, {}) == hashlib.sha256(body).hexdigest()     # sin huella previa: se registra


# ─── transiciones con excepción histórica acotada ──────────────────────────

def _season(teams):
    return pd.DataFrame([(pd.Timestamp("2000-01-01"), a, b, 1, 0) for a, b in itertools.permutations(teams, 2)],
                        columns=["date", "home", "away", "hg", "ag"])


def test_historical_exception_allows_four_entries_only_for_that_pair():
    base = list("ABCDEFGH")
    four = list("ABCD") + list("WXYZ")
    with pytest.raises(ValueError):
        ingest.check_transitions({"1011": _season(base), "1112": _season(four)})              # otra temporada: 4 no vale
    ingest.check_transitions({"9899": _season(base), "9900": _season(four)})                  # la excepción documentada
    five = list("ABC") + list("VWXYZ")
    with pytest.raises(ValueError):
        ingest.check_transitions({"9899": _season(base), "9900": _season(five)})              # ni siquiera con 5


# ─── construcción del histórico de extremo a extremo ───────────────────────

CLUBS_M = ["Ath Madrid", "Betis", "Barcelona", "Valencia"]                 # nombres del espejo
CLUBS_C = ["Club Atlético de Madrid", "Real Betis Balompié", "FC Barcelona", "Valencia CF"]   # canónicos


def _season_rows(clubs, start, extra=()):
    rows, i = [], 0
    for a, b in itertools.permutations(clubs, 2):
        d = (dt.date.fromisoformat(start) + dt.timedelta(days=3 * i)).isoformat()
        rows.append((d, a, b, (i * 2) % 4, (i * 3) % 3))
        i += 1
    return rows + list(extra)


def _bodies_and_refs():
    old = _season_rows(CLUBS_M, "1999-09-05")                                       # solo espejo
    ov = _season_rows(CLUBS_M, "2020-09-05")                                        # solape con openfootball
    ref_rows = [(d, CLUBS_C[CLUBS_M.index(h)], CLUBS_C[CLUBS_M.index(a)], g, c) for d, h, a, g, c in ov][:-2]
    extra = [(ov[-2][0], *ov[-2][1:]), (ov[-1][0], *ov[-1][1:])]                    # 2 filas solo en el espejo (plausibles)
    zero = _season_rows(CLUBS_M, "2021-09-05")
    zero_ref = [(d, CLUBS_C[CLUBS_M.index(h)], CLUBS_C[CLUBS_M.index(a)], g, c) for d, h, a, g, c in zero][:-3]
    zero_mir = zero_ref and [(d, h, a, g, c) for d, h, a, g, c in zero[:-3]] + [(d, h, a, 0, 0) for d, h, a, g, c in zero[-3:]]
    zero_mir = [(d, CLUBS_M[CLUBS_C.index(h)] if h in CLUBS_C else h, CLUBS_M[CLUBS_C.index(a)] if a in CLUBS_C else a, g, c)
                for d, h, a, g, c in zero_mir]
    bodies = {"9900": mirror_csv(old).encode(), "2021": mirror_csv(ov).encode(), "2122": mirror_csv(zero_mir).encode()}
    refs = {"2021": df_of(ref_rows), "2122": df_of(zero_ref)}
    return bodies, refs


def test_build_history_merges_quarantines_and_reports():
    bodies, refs = _bodies_and_refs()
    seasons, rep = sources.build_history(bodies, refs, TODAY)
    assert list(seasons) == ["9900", "2021", "2122"]                                  # orden cronológico (cambio de siglo)
    assert rep["seasons"]["9900"]["source"] == "mirror"
    assert rep["seasons"]["2021"] == {"source": "merged", "reference": 10, "added_from_mirror": 2, "quarantined": 0}
    assert len(seasons["2021"]) == 12                                                  # 10 de referencia + 2 respaldadas
    assert rep["seasons"]["2122"]["quarantined"] == 3 and len(seasons["2122"]) == 9    # 3 ceros de relleno descartados
    assert len(rep["quarantined"]) == 3 and all(q["hg"] == 0 and q["ag"] == 0 for q in rep["quarantined"])
    assert set(seasons["9900"]["home"]) <= set(CLUBS_C)                                # alias aplicados al histórico
    assert rep["aliases_verified_against_data"] >= 3 and len(rep["mirror_sha256"]) == 3


def test_build_history_fails_loudly_on_changed_file_unreviewed_similar_names_and_bad_alias():
    bodies, refs = _bodies_and_refs()
    _, rep = sources.build_history(bodies, refs, TODAY)
    tampered = dict(rep["mirror_sha256"], **{"2021": "0" * 64})
    with pytest.raises(sources.SourceChanged, match="2021"):
        sources.build_history(bodies, refs, TODAY, expected_hashes=tampered)
    sources.build_history(bodies, refs, TODAY, expected_hashes=tampered, accept_changes=True)
    typo = dict(bodies, **{"9900": mirror_csv([("1999-09-05", "Numancia", "Barcelona", 1, 0), ("1999-09-12", "Numancía", "Valencia", 2, 2)]).encode()})
    with pytest.raises(ValueError, match="parecidos"):
        sources.build_history(typo, refs, TODAY)            # dos escrituras del mismo club sin alias unificado
