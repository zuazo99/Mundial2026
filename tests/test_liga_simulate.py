"""Simulación del resto de la temporada de La Liga (Monte Carlo sobre XG fijo)."""
import json

import numpy as np
import pandas as pd
import pytest
from scipy.stats import poisson

from liga import ingest, model as lm, simulate as sim
from liga_synth import TEAMS, make_seasons


# ─── desempates de La Liga ─────────────────────────────────────────────────

def _matrices(n, results):
    """results: {(local, visitante): (goles_local, goles_visitante)} → (hg, ag) n×n."""
    hg, ag = np.zeros((n, n), dtype=int), np.zeros((n, n), dtype=int)
    for (h, a), (gh, ga) in results.items():
        hg[h, a], ag[h, a] = gh, ga
    return hg, ag


A, B, C, D = 0, 1, 2, 3


def test_two_way_tie_head_to_head_beats_overall_goal_difference():
    # A y B empatan a 12; B ganó los dos directos pero A tiene mucha mejor diferencia general.
    res = {(A, B): (0, 1), (B, A): (1, 0),
           (A, C): (5, 0), (C, A): (0, 5), (A, D): (5, 0), (D, A): (0, 5),
           (B, C): (1, 0), (C, B): (0, 1), (B, D): (0, 1), (D, B): (1, 0),
           (C, D): (0, 0), (D, C): (0, 0)}
    hg, ag = _matrices(4, res)
    pts, gf, ga = sim._table(hg, ag)
    assert pts[A] == pts[B] == 12 and (gf - ga)[A] > (gf - ga)[B]      # el desempate general diría A
    assert list(sim.final_order(hg, ag)) == [B, A, D, C]               # el directo manda


def test_three_way_tie_uses_mini_league_before_overall_goal_difference():
    # A, B, C empatan a 12. Mini-liga entre ellos: A (+4 DG), C (0), B (-4). DG general: B > A > C.
    res = {(A, B): (3, 0), (B, A): (0, 3), (B, C): (1, 0), (C, B): (0, 1),
           (C, A): (1, 0), (A, C): (0, 1),
           (A, D): (1, 0), (D, A): (0, 1), (B, D): (20, 0), (D, B): (0, 1),
           (C, D): (1, 0), (D, C): (0, 1)}
    hg, ag = _matrices(4, res)
    pts, gf, ga = sim._table(hg, ag)
    assert pts[A] == pts[B] == pts[C] == 12 and pts[D] == 0
    gd = gf - ga
    assert gd[B] > gd[A] > gd[C]
    assert list(sim.final_order(hg, ag)) == [A, C, B, D]


def test_full_head_to_head_tie_falls_back_to_overall_goal_difference():
    res = {(0, 1): (1, 0), (1, 0): (1, 0),                      # directos idénticos
           (0, 2): (5, 0), (2, 0): (0, 5), (1, 2): (1, 0), (2, 1): (0, 1)}
    hg, ag = _matrices(3, res)
    assert list(sim.final_order(hg, ag)) == [0, 1, 2]           # 0 tiene mejor DG general
    res[(0, 2)], res[(2, 0)] = (1, 0), (0, 1)                   # ahora 1 tiene igual DG; gf decide
    hg, ag = _matrices(3, res)
    assert sim.final_order(hg, ag)[2] == 2


# ─── muestreador ───────────────────────────────────────────────────────────

def test_sampler_means_match_xg_without_rho():
    rng = np.random.default_rng(0)
    hg, ag = sim.sample_scores(np.array([1.6, 0.8]), np.array([1.1, 2.0]), 0.0, 200_000, rng)
    assert hg.shape == (200_000, 2)
    assert hg.mean(axis=0) == pytest.approx([1.6, 0.8], abs=0.02)
    assert ag.mean(axis=0) == pytest.approx([1.1, 2.0], abs=0.02)


def test_sampler_low_score_frequencies_match_dixon_coles_pmf():
    lam, mu, rho = 1.4, 1.0, -0.1
    rng = np.random.default_rng(1)
    hg, ag = sim.sample_scores(np.array([lam]), np.array([mu]), rho, 400_000, rng)
    k = np.arange(11)
    joint = np.outer(poisson.pmf(k, lam), poisson.pmf(k, mu))
    joint[0, 0] *= 1 - lam * mu * rho
    joint[1, 0] *= 1 + mu * rho
    joint[0, 1] *= 1 + lam * rho
    joint[1, 1] *= 1 - rho
    joint /= joint.sum()
    for (x, y) in [(0, 0), (1, 0), (0, 1), (1, 1), (2, 1)]:
        assert ((hg[:, 0] == x) & (ag[:, 0] == y)).mean() == pytest.approx(joint[x, y], abs=0.004)


def test_negative_rho_makes_draws_more_likely_than_independent_poisson():
    rng = np.random.default_rng(2)
    d0 = (lambda h, a: (h == a).mean())(*sim.sample_scores(np.array([1.4]), np.array([1.1]), 0.0, 300_000, rng))
    d1 = (lambda h, a: (h == a).mean())(*sim.sample_scores(np.array([1.4]), np.array([1.1]), -0.1, 300_000, rng))
    assert d1 > d0 + 0.01


# ─── simulación de temporada ───────────────────────────────────────────────

@pytest.fixture(scope="module")
def world():
    seasons = make_seasons()
    long = ingest.to_long(ingest.add_elo(seasons))
    last = list(seasons)[-1]
    pending = ingest.pending_fixtures(seasons[last])
    elo_now = ingest.final_elos(seasons)
    mdl = lm.train_liga(long)
    xg = lm.predict_fixtures(mdl, lm.team_state(long, mdl, elo_now), pending)
    played = seasons[last].rename(columns={"hg": "hg", "ag": "ag"})[["home", "away", "hg", "ag"]]
    return dict(seasons=seasons, long=long, pending=pending, xg=xg, played=played, rho=mdl.rho, last=last, elo_now=elo_now)


def _run(world, n_sims=2000, seed=7, **kw):
    return sim.simulate_season(TEAMS, world["played"], world["xg"], world["rho"], n_sims, seed, **kw)


def test_probability_invariants(world):
    out = _run(world)
    t = out["teams"]
    assert sum(v["p_champion"] for v in t.values()) == pytest.approx(1.0)
    assert sum(v["p_champions"] for v in t.values()) == pytest.approx(4.0)
    assert sum(v["p_europa"] for v in t.values()) == pytest.approx(2.0)
    assert sum(v["p_relegation"] for v in t.values()) == pytest.approx(3.0)
    pos = np.array([v["position_dist"] for v in t.values()])
    assert pos.shape == (len(TEAMS), len(TEAMS))
    assert pos.sum(axis=1) == pytest.approx(np.ones(len(TEAMS)))     # cada equipo acaba en alguna posición
    assert pos.sum(axis=0) == pytest.approx(np.ones(len(TEAMS)))     # cada posición la ocupa alguien
    for v in t.values():
        assert v["exp_points"] >= v["points_now"]


def test_same_seed_same_result_and_different_seed_differs(world):
    assert _run(world, seed=3) == _run(world, seed=3)
    assert _run(world, seed=3) != _run(world, seed=4)


def test_metadata_documents_assumptions(world):
    meta = _run(world, n_sims=500, seed=11)["meta"]
    assert meta["n_sims"] == 500 and meta["seed"] == 11
    assert meta["xg_frozen"] is True
    assert meta["rho"] == pytest.approx(world["rho"])
    assert meta["zones"] == {"champions": 4, "europa": [5, 6], "relegation": 3}
    assert meta["pending_matches"] == len(world["xg"])


def test_strongest_team_is_more_likely_champion_and_weakest_to_be_relegated(world):
    t = _run(world)["teams"]
    strongest, weakest = TEAMS[-1], TEAMS[0]
    assert t[strongest]["p_champion"] > t[weakest]["p_champion"]
    assert t[weakest]["p_relegation"] > t[strongest]["p_relegation"]
    assert t[strongest]["exp_position"] < t[weakest]["exp_position"]


def test_finished_season_is_deterministic(world):
    full = world["seasons"][list(world["seasons"])[-2]][["home", "away", "hg", "ag"]]
    empty_xg = pd.DataFrame(columns=["home", "away", "xg_home", "xg_away"])
    t = sim.simulate_season(TEAMS, full, empty_xg, world["rho"], 200, 0)["teams"]
    assert max(v["p_champion"] for v in t.values()) == 1.0
    assert sum(v["p_champion"] for v in t.values()) == 1.0
    assert all(max(v["position_dist"]) == 1.0 for v in t.values())


def test_custom_zones_are_respected(world):
    out = _run(world, n_sims=300, zones={"champions": 3, "europa": (4, 5), "relegation": 2})
    assert sum(v["p_champions"] for v in out["teams"].values()) == pytest.approx(3.0)
    assert sum(v["p_relegation"] for v in out["teams"].values()) == pytest.approx(2.0)


# ─── validación de entradas ────────────────────────────────────────────────

def test_rejects_pending_matches_already_played(world):
    bad = pd.concat([world["xg"], world["played"].head(1).assign(xg_home=1.0, xg_away=1.0)[world["xg"].columns]])
    with pytest.raises(ValueError, match="jugado"):
        sim.simulate_season(TEAMS, world["played"], bad, 0.0, 10, 0)


def test_rejects_incomplete_calendar(world):
    with pytest.raises(ValueError, match="calendario"):
        sim.simulate_season(TEAMS, world["played"], world["xg"].iloc[1:], 0.0, 10, 0)


def test_rejects_unknown_team(world):
    bad = world["xg"].copy()
    bad.loc[0, "home"] = "Fantasma"
    with pytest.raises(ValueError, match="equipo"):
        sim.simulate_season(TEAMS, world["played"], bad, 0.0, 10, 0)


# ─── CLI / run ─────────────────────────────────────────────────────────────

def _write_artifacts(tmp_path, world, code=None, xg_code=None):
    data, models = tmp_path / "liga", tmp_path / "models"
    data.mkdir(), models.mkdir()
    world["long"].to_csv(data / "df_liga.csv", index=False)
    world["pending"].to_csv(data / f"pending_{code or world['last']}.csv", index=False)
    world["xg"].to_csv(data / f"xg_pending_{xg_code or world['last']}.csv", index=False)
    (models / "rho_liga.json").write_text(json.dumps({"variant": "liga", "rho": world["rho"]}))
    return str(data), str(models), str(tmp_path / "results")


def test_run_writes_probabilities_in_given_dir(tmp_path, world):
    data, models, results = _write_artifacts(tmp_path, world)
    sim.run(data, models, results, n_sims=300, seed=1)
    out = json.load(open(tmp_path / "results" / "liga_probabilities.json"))
    assert out["meta"]["season"] == world["last"]
    assert set(out["teams"]) == set(TEAMS)
    assert out["meta"]["data_as_of"] == str(world["long"].loc[world["long"]["season"] == world["last"], "date"].max())[:10]


def test_run_without_model_artifacts_points_to_model_py(tmp_path):
    with pytest.raises(SystemExit, match="model.py"):
        sim.run(str(tmp_path), str(tmp_path / "m"), str(tmp_path / "r"))


def test_run_aborts_when_xg_belongs_to_another_season(tmp_path, world):
    data, models, results = _write_artifacts(tmp_path, world, xg_code="9900")
    with pytest.raises(SystemExit, match="model.py"):
        sim.run(data, models, results, n_sims=10)


def test_run_aborts_when_pending_is_stale_vs_ingested_season(tmp_path, world):
    data, models, results = _write_artifacts(tmp_path, world, code="9900", xg_code="9900")
    with pytest.raises(SystemExit, match="ingest"):
        sim.run(data, models, results, n_sims=10)


def test_run_aborts_without_rho_instead_of_assuming_zero(tmp_path, world):
    data, models, results = _write_artifacts(tmp_path, world)
    (tmp_path / "models" / "rho_liga.json").unlink()
    with pytest.raises(SystemExit, match="rho"):
        sim.run(data, models, results, n_sims=10)


def test_full_chain_from_football_data_csv_to_probabilities(tmp_path):
    """SP1_*.csv (formato football-data) → ingest → model → simulate, con los contratos reales entre fases."""
    seasons = make_seasons()
    raw = tmp_path / "raw"
    raw.mkdir()
    for code, df in seasons.items():
        pd.DataFrame({"Div": "SP1", "Date": df["date"].dt.strftime("%d/%m/%Y"), "HomeTeam": df["home"],
                      "AwayTeam": df["away"], "FTHG": df["hg"], "FTAG": df["ag"]}).to_csv(raw / f"SP1_{code}.csv", index=False)
    data, models, results = tmp_path / "liga", tmp_path / "models", tmp_path / "results"

    ingest.build_dataset(str(raw), str(data))
    lm.run(str(data), str(models), str(results))
    out = sim.run(str(data), str(models), str(results), n_sims=300, seed=5)

    last = list(seasons)[-1]
    saved = json.load(open(results / "liga_probabilities.json"))
    assert saved["meta"]["season"] == last == out["meta"]["season"]
    assert set(saved["teams"]) == set(TEAMS)
    assert sum(v["p_champion"] for v in saved["teams"].values()) == pytest.approx(1.0)
    assert saved["meta"]["pending_matches"] == len(TEAMS) * (len(TEAMS) - 1) - len(seasons[last])
