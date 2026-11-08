"""Los códigos de temporada son texto ('9899', '9900', '0001'): el orden debe ser cronológico, no alfabético."""
import json

import pandas as pd
import pytest

from liga import ingest, model as lm, simulate as sim
from liga_synth import TEAMS, make_seasons

CODES = ["9798", "9899", "9900", "0001", "0102"]          # cruzan el cambio de siglo


def y2k_seasons():
    return {new: df for new, (_, df) in zip(CODES, make_seasons().items())}


def test_season_key_is_chronological_across_the_century_change():
    keys = [ingest.season_key(c) for c in ["9394", "9899", "9900", "0001", "1617", "2627"]]
    assert keys == sorted(keys) and len(set(keys)) == 6


def test_load_seasons_orders_chronologically(tmp_path):
    for code in ["2627", "9394", "0001", "9900"]:
        (tmp_path / f"SP1_{code}.csv").write_text("Div,Date,HomeTeam,AwayTeam,FTHG,FTAG\nSP1,15/08/2000,A,B,1,0\n")
    assert list(ingest.load_seasons(str(tmp_path))) == ["9394", "9900", "0001", "2627"]


def test_train_liga_drops_the_chronologically_first_season():
    seasons = y2k_seasons()
    long = ingest.to_long(ingest.add_elo(seasons))
    m = lm.train_liga(long)
    assert m.train_seasons == ["9899", "9900", "0001", "0102"]          # '9798' es la de arranque en frío


def test_holdout_default_test_season_is_the_latest_complete_one():
    seasons = y2k_seasons()
    long = ingest.to_long(ingest.add_elo(seasons))
    v = lm.holdout_validation(long, test_season="0001")
    assert v["train_seasons"] == ["9899", "9900"]                      # solo las anteriores en el tiempo


def test_simulate_picks_the_chronologically_last_season(tmp_path):
    seasons = y2k_seasons()
    last = CODES[-1]
    long = ingest.to_long(ingest.add_elo(seasons))
    pend = ingest.pending_fixtures(seasons[last])
    xg = pend.assign(xg_home=1.4, xg_away=1.1)
    data, models = tmp_path / "liga", tmp_path / "models"
    data.mkdir(), models.mkdir()
    long.to_csv(data / "df_liga.csv", index=False)
    pend.to_csv(data / f"pending_{last}.csv", index=False)
    xg.to_csv(data / f"xg_pending_{last}.csv", index=False)
    (models / "rho_liga.json").write_text(json.dumps({"rho": -0.02}))
    season, teams, played, xg_in, rho, as_of = sim.load_inputs(str(data), str(models))
    assert season == last and set(teams) == set(TEAMS)


def test_train_from_fixes_training_rows_independently_of_the_elo_history():
    seasons = y2k_seasons()
    long = ingest.to_long(ingest.add_elo(seasons))
    default = lm.train_liga(long)
    assert default.train_seasons == ["9899", "9900", "0001", "0102"]
    explicit = lm.train_liga(long, train_from="9899")                       # igual que el valor por defecto
    assert explicit.train_seasons == default.train_seasons and explicit.n_train_rows == default.n_train_rows
    later = lm.train_liga(long, train_from="0001")                          # menos filas de entrenamiento, misma historia
    assert later.train_seasons == ["0001", "0102"] and later.n_train_rows < default.n_train_rows
    earliest = lm.train_liga(long, train_from="9798")                       # incluye también la de arranque en frío
    assert earliest.train_seasons == CODES and earliest.n_train_rows > default.n_train_rows


def test_train_from_rejects_unknown_or_empty_training_window():
    long = ingest.to_long(ingest.add_elo(y2k_seasons()))
    with pytest.raises(ValueError, match="train_from"):
        lm.train_liga(long, train_from="2627")                              # posterior a todos los datos
