"""Ingesta de La Liga: parser, formato largo, ELO propio, calendario y enlace con el modelo."""
import os

import numpy as np
import pandas as pd
import pytest

from liga import ingest

FIX = os.path.join(os.path.dirname(__file__), "fixtures")


@pytest.fixture(scope="module")
def seasons():
    return ingest.load_seasons(FIX)


# ─── parser ────────────────────────────────────────────────────────────────

def test_read_season_handles_bom_blank_rows_and_both_date_formats():
    df = ingest.read_season(os.path.join(FIX, "SP1_2122.csv"))
    assert len(df) == 12                                  # fila vacía descartada
    assert df["date"].min() == pd.Timestamp("2021-08-15")  # dd/mm/yy
    assert (df["date"] == pd.Timestamp("2021-08-22")).sum() == 2  # dd/mm/yyyy
    assert list(df.columns) == ["date", "home", "away", "hg", "ag"]
    assert df["hg"].dtype.kind == "i" and df["ag"].dtype.kind == "i"


def test_read_season_reads_latin1_and_drops_duplicate_match():
    df = ingest.read_season(os.path.join(FIX, "SP1_2223.csv"))
    assert len(df) == 6  # 7 filas con 1 duplicado exacto


def test_load_seasons_sorted_and_keyed_by_code(seasons):
    assert list(seasons) == ["2122", "2223"]


def test_read_season_rejects_missing_columns(tmp_path):
    bad = tmp_path / "SP1_9999.csv"
    bad.write_text("Date,HomeTeam\n01/01/20,A\n")
    with pytest.raises(ValueError, match="columnas"):
        ingest.read_season(str(bad))


# ─── ELO propio ────────────────────────────────────────────────────────────

def test_elo_is_pre_match_and_starts_at_initial(seasons):
    m = ingest.add_elo(seasons)
    first = m[m["season"] == "2122"].sort_values("date").iloc[0]
    assert first["elo_home"] == ingest.ELO_INIT and first["elo_away"] == ingest.ELO_INIT


def test_elo_update_is_zero_sum_within_a_season(seasons):
    final = ingest.final_elos({"2122": seasons["2122"]})
    assert len(final) == 4
    assert sum(final.values()) == pytest.approx(4 * ingest.ELO_INIT)


def test_promoted_team_starts_at_mean_elo_of_relegated(seasons):
    final_2122 = ingest.final_elos({"2122": seasons["2122"]})
    m = ingest.add_elo(seasons)
    new_first = m[(m["season"] == "2223") & ((m["home"] == "Epsilón") | (m["away"] == "Epsilón"))].sort_values("date").iloc[0]
    new_elo = new_first["elo_home"] if new_first["home"] == "Epsilón" else new_first["elo_away"]
    assert new_elo == pytest.approx(final_2122["Delta"])   # único descendido
    # los que continúan arrastran su ELO
    alpha = m[(m["season"] == "2223") & (m["home"] == "Alpha")].sort_values("date").iloc[0]["elo_home"]
    assert alpha == pytest.approx(final_2122["Alpha"])


def test_stronger_team_gains_less_for_expected_win():
    gain_fav = ingest.elo_delta(1700, 1500, 1, 0)
    gain_dog = ingest.elo_delta(1500, 1700, 0, 1)   # el local débil pierde
    assert gain_fav > 0 > gain_dog
    up = ingest.elo_delta(1500, 1700, 1, 0)         # upset del local
    assert up > gain_fav


# ─── formato largo ─────────────────────────────────────────────────────────

def test_to_long_two_rows_per_match_with_mirrored_goals(seasons):
    m = ingest.add_elo(seasons)
    long = ingest.to_long(m)
    assert len(long) == 2 * len(m)
    assert set(long.columns) >= {"date", "team", "opponent", "goals", "goals_conceded",
                                 "is_home", "elo", "opponent_elo", "tournament"}
    assert long["is_home"].isin([0, 1]).all()
    assert long.groupby("date")["is_home"].sum().eq(long.groupby("date").size() / 2).all()
    assert (long["tournament"] == "La Liga").all()
    # simetría: lo que mete uno es lo que encaja el otro (cruce por partido)
    h = long[long["is_home"] == 1]
    a = long[long["is_home"] == 0]
    j = h.merge(a, left_on=["date", "team", "opponent"], right_on=["date", "opponent", "team"],
                suffixes=("_h", "_a"))
    assert len(j) == len(m)
    assert (j["goals_h"] == j["goals_conceded_a"]).all()
    assert (j["goals_conceded_h"] == j["goals_a"]).all()
    assert (j["elo_h"] == j["opponent_elo_a"]).all()


def test_long_output_feeds_the_model_pipeline(seasons):
    import xg_preds as xp
    long = ingest.to_long(ingest.add_elo(seasons))
    df, *_ = xp.engineer_features(long.sort_values(["team", "date"]).reset_index(drop=True))
    assert set(xp.FEATURES) <= set(df.columns)
    assert len(df) == len(long)


# ─── calendario pendiente ──────────────────────────────────────────────────

def test_pending_fixtures_complete_the_double_round_robin(seasons):
    cur = seasons["2223"]
    pend = ingest.pending_fixtures(cur)
    teams = sorted(set(cur["home"]) | set(cur["away"]))
    n = len(teams)
    assert n == 4
    assert len(cur) + len(pend) == n * (n - 1)            # 12 = 4*3
    assert (pend["home"] != pend["away"]).all()
    played = set(zip(cur["home"], cur["away"]))
    assert not played & set(zip(pend["home"], pend["away"]))
    games = pd.concat([pend["home"], pend["away"], cur["home"], cur["away"]]).value_counts()
    assert (games == 2 * (n - 1)).all()                   # cada equipo, 2*(n-1) partidos


def test_pending_fixtures_empty_for_finished_season(seasons):
    assert ingest.pending_fixtures(seasons["2122"]).empty


def test_build_dataset_end_to_end(tmp_path):
    import shutil
    raw = tmp_path / "raw"
    raw.mkdir()
    for f in ("SP1_2122.csv", "SP1_2223.csv"):
        shutil.copy(os.path.join(FIX, f), raw / f)
    out = tmp_path / "out"
    ingest.build_dataset(str(raw), str(out))
    long = pd.read_csv(out / "df_liga.csv", parse_dates=["date"])
    assert len(long) == 2 * (12 + 6)
    pend = pd.read_csv(out / "pending_2223.csv")
    assert len(pend) == 12 - 6
    assert (out / "elo_actual.csv").exists()


def test_build_dataset_without_raw_files_explains_how_to_get_them(tmp_path):
    with pytest.raises(SystemExit, match="download_liga"):
        ingest.build_dataset(str(tmp_path), str(tmp_path / "out"))
