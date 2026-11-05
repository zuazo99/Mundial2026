"""Modelo de La Liga con una liga sintética de fuerzas conocidas.

Los datos pasan por la ingesta real (temporadas → add_elo → to_long), así que
también se prueba el enlace con la fase anterior. Nada se escribe fuera de tmp_path.
"""
import itertools
import json
import os

import numpy as np
import pandas as pd
import pytest

from liga import ingest, model as lm

N_TEAMS, N_SEASONS = 12, 5
TEAMS = [f"T{i:02d}" for i in range(N_TEAMS)]
STRENGTH = dict(zip(TEAMS, np.linspace(-1.0, 1.0, N_TEAMS)))  # T11 fuerte, T00 débil


def _round_robin_rounds(teams):
    """Método del círculo: n-1 rondas, cada equipo juega exactamente una vez por ronda."""
    ts = list(teams)
    n = len(ts)
    rounds = []
    for _ in range(n - 1):
        rounds.append([(ts[i], ts[n - 1 - i]) for i in range(n // 2)])
        ts = [ts[0]] + [ts[-1]] + ts[1:-1]
    return rounds


def make_seasons(seed=0, last_fraction=0.6):
    """Doble vuelta semanal con calendario válido; la última temporada solo se ha jugado en parte."""
    rng = np.random.default_rng(seed)
    seasons = {}
    for k in range(N_SEASONS):
        first_leg = _round_robin_rounds(TEAMS)
        rounds = first_leg + [[(a, h) for h, a in r] for r in first_leg]
        start = pd.Timestamp(2021 + k, 8, 15)
        rows = []
        for ri, matches in enumerate(rounds):
            for h, a in matches:
                if rng.random() < 0.5 and ri < len(first_leg):  # alterna localías de la ida
                    h, a = a, h
                d = STRENGTH[h] - STRENGTH[a]
                lam_h, lam_a = 1.45 * np.exp(0.55 * d) * 1.15, 1.45 * np.exp(-0.55 * d) / 1.1
                rows.append((start + pd.Timedelta(days=7 * ri), h, a,
                             int(rng.poisson(lam_h)), int(rng.poisson(lam_a))))
        df = pd.DataFrame(rows, columns=["date", "home", "away", "hg", "ag"])
        if k == N_SEASONS - 1:
            df = df.iloc[: int(len(df) * last_fraction)]
        seasons[f"{21 + k}{22 + k}"] = df.reset_index(drop=True)
    return seasons


@pytest.fixture(scope="module")
def world():
    seasons = make_seasons()
    matches = ingest.add_elo(seasons)
    long = ingest.to_long(matches)
    last = list(seasons)[-1]
    pending = ingest.pending_fixtures(seasons[last])
    elo_now = ingest.final_elos(seasons)
    model = lm.train_liga(long)
    state = lm.team_state(long, model, elo_now)
    return dict(seasons=seasons, long=long, pending=pending, elo_now=elo_now, model=model, state=state, last=last)


def test_first_season_is_not_used_to_fit(world):
    m, seasons = world["model"], world["seasons"]
    first = list(seasons)[0]
    assert first not in m.train_seasons
    assert m.train_seasons == list(seasons)[1:]
    expected_rows = 2 * sum(len(df) for c, df in seasons.items() if c != first)
    assert m.n_train_rows == expected_rows


def test_decay_reference_is_latest_training_date(world):
    assert world["model"].decay_ref == world["long"]["date"].max()


def test_team_state_uses_current_elo_and_covers_current_teams(world):
    st = world["state"]
    assert set(st.index) == set(world["elo_now"])
    for t in st.index:
        assert st.loc[t, "elo"] == pytest.approx(world["elo_now"][t])
    assert not st[["gf_prom_5", "gc_prom_5", "PCA_1", "PCA_2"]].isna().any().any()


def test_stronger_team_at_home_has_more_xg_than_the_reverse(world):
    best, worst = TEAMS[-1], TEAMS[0]
    out = lm.predict_fixtures(world["model"], world["state"],
                              pd.DataFrame({"home": [best, worst], "away": [worst, best]}))
    assert out.loc[0, "xg_home"] > out.loc[0, "xg_away"]            # fuerte en casa gana en XG
    assert out.loc[0, "xg_home"] > out.loc[1, "xg_away"]             # fuerte como local > fuerte como visitante
    assert out.loc[1, "xg_home"] < out.loc[1, "xg_away"]


def test_home_side_has_more_xg_on_average(world):
    out = lm.predict_fixtures(world["model"], world["state"], world["pending"])
    assert out["xg_home"].mean() > out["xg_away"].mean()


def test_pending_prediction_is_complete_and_positive(world):
    out = lm.predict_fixtures(world["model"], world["state"], world["pending"])
    assert len(out) == len(world["pending"])
    assert list(out["home"]) == list(world["pending"]["home"])
    assert not out[["xg_home", "xg_away"]].isna().any().any()
    assert (out[["xg_home", "xg_away"]] > 0).all().all()


def test_holdout_beats_constant_baseline_without_leakage(world):
    seasons = {c: df for c, df in make_seasons(seed=1).items()}
    # temporada de test = la última completa (la parcial queda fuera)
    v = lm.holdout_validation(ingest.to_long(ingest.add_elo(seasons)))
    assert v["test_season"] == list(seasons)[-2]
    assert v["model"]["poisson_deviance"] < v["baseline_constant"]["poisson_deviance"]
    assert v["model"]["poisson_deviance"] < v["baseline_home_away"]["poisson_deviance"]
    assert v["model"]["mae"] < v["baseline_constant"]["mae"]
    assert v["n_test_rows"] == 2 * N_TEAMS * (N_TEAMS - 1)


def test_holdout_needs_enough_seasons():
    seasons = dict(list(make_seasons().items())[:2])
    with pytest.raises(ValueError, match="temporadas"):
        lm.holdout_validation(ingest.to_long(ingest.add_elo(seasons)))


def test_run_writes_artifacts_only_inside_given_dirs(tmp_path, world):
    data_dir, model_dir, results_dir = tmp_path / "liga", tmp_path / "models", tmp_path / "results"
    data_dir.mkdir()
    world["long"].to_csv(data_dir / "df_liga.csv", index=False)
    world["pending"].to_csv(data_dir / f"pending_{world['last']}.csv", index=False)
    pd.Series(world["elo_now"], name="elo").rename_axis("team").to_csv(data_dir / "elo_actual.csv")

    lm.run(str(data_dir), str(model_dir), str(results_dir))

    assert (model_dir / "xgb_liga.json").exists()
    rho = json.load(open(model_dir / "rho_liga.json"))
    assert -0.2 <= rho["rho"] <= 0.2
    xg = pd.read_csv(data_dir / f"xg_pending_{world['last']}.csv")
    assert len(xg) == len(world["pending"]) and not xg.isna().any().any()
    assert (results_dir / "validation_liga.json").exists()


def test_run_without_ingested_data_fails_with_clear_message(tmp_path):
    with pytest.raises(SystemExit, match="ingest"):
        lm.run(str(tmp_path), str(tmp_path / "m"), str(tmp_path / "r"))


def test_valid_calendar_has_one_match_per_team_per_date():
    for df in make_seasons().values():
        long = ingest.to_long(ingest.add_elo({"x": df}))
        assert not long.duplicated(["date", "team"]).any()


def test_train_rejects_team_playing_twice_on_the_same_date(world):
    bad = pd.concat([world["long"], world["long"].iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="misma fecha"):
        lm.train_liga(bad)


def test_country_like_club_name_does_not_leak_confederation_into_training():
    """'Andorra' está en la lista UEFA de selecciones: el entrenamiento no debe verlo."""
    seasons = make_seasons()
    renamed = {c: df.replace({"T05": "Andorra"}) for c, df in seasons.items()}
    long = ingest.to_long(ingest.add_elo(renamed))
    seen = {}
    orig = lm._fit_xgb

    def spy(df_train, feat_means):
        seen["confed"] = set(df_train["confed"]) | set(df_train["rival_confed"])
        seen["tournament_num"] = set(df_train["tournament_num"])
        return orig(df_train, feat_means)

    try:
        lm._fit_xgb = spy
        lm.train_liga(long)
        assert seen == {"confed": {0}, "tournament_num": {1}}
        seen.clear()
        lm.holdout_validation(long)
        assert seen == {"confed": {0}, "tournament_num": {1}}
    finally:
        lm._fit_xgb = orig
