"""Motor de simulación: Team, Match (Dixon-Coles) y desempates de Group."""
import random

import pandas as pd
import pytest

import clases_simulacion as cs


def _team(name, group="A"):
    df = pd.DataFrame([dict(team=name, gf_prom_5=1, gc_prom_5=1, gf_prom_15=1, gc_prom_15=1,
                            elo_prom_5=1800, PCA_1=0.0, PCA_2=0.0, confed=1)])
    return cs.Team(name, 1800, group, df)


@pytest.fixture(autouse=True)
def _reset_rho():
    cs.set_dc_rho(0.0)
    yield
    cs.set_dc_rho(0.0)


def test_team_update_result_points_and_goals():
    t = _team("A")
    t.update_result(2, 0)
    t.update_result(1, 1)
    t.update_result(0, 3)
    assert (t.points, t.gf, t.dg) == (4, 3, -1)


def test_tau_is_identity_without_rho():
    m = cs.Match(_team("A"), _team("B"), 1.4, 1.1)
    assert all(m._tau(a, b) == 1.0 for a in range(4) for b in range(4))


def test_tau_negative_rho_boosts_draws_and_downweights_one_nil():
    cs.set_dc_rho(-0.1)
    m = cs.Match(_team("A"), _team("B"), 1.4, 1.1)
    assert m._tau(0, 0) > 1 and m._tau(1, 1) > 1
    assert m._tau(1, 0) < 1 and m._tau(0, 1) < 1
    assert m._tau(2, 1) == 1.0 and m._tau(3, 3) == 1.0


def test_multiplier_trailing_team_attacks_more():
    m = cs.Match(_team("A"), _team("B"), 1.0, 1.0)
    m1, m2 = m.get_multiplier(60, 1, 0)   # A gana → B (mult2) aprieta
    assert m2 > m1
    m1, m2 = m.get_multiplier(60, 0, 1)
    assert m1 > m2


def test_simulate_common_updates_both_teams_in_group_stage(capsys):
    random.seed(1)
    a, b = _team("A"), _team("B")
    res = cs.Match(a, b, 1.8, 0.6).simulate_common()
    ga, gb = res[a], res[b]
    assert isinstance(ga, int) and isinstance(gb, int)
    assert a.gf == ga and b.gf == gb
    assert a.points + b.points in (2, 3)


def test_knockout_match_always_has_a_winner(capsys):
    random.seed(7)
    for _ in range(20):
        a, b = _team("A"), _team("B")
        res = cs.Match(a, b, 1.0, 1.0, ko=True).simulate_common()
        assert res[a] != res[b]


def test_penalties_never_tie():
    random.seed(3)
    m = cs.Match(_team("A"), _team("B"), 1.0, 1.0)
    for _ in range(200):
        p1, p2 = m.simulate_penalties()
        assert p1 != p2


def _group_with(results):
    """Group sin I/O con resultados dados: [(A,B,ga,gb), ...]."""
    teams = {n: _team(n) for n in "ABCD"}
    g = cs.Group.__new__(cs.Group)
    g.s1, g.s2, g.s3, g.s4 = (teams[n] for n in "ABCD")
    g.letter = "A"
    g.results = []
    for a, b, ga, gb in results:
        teams[a].update_result(ga, gb)
        teams[b].update_result(gb, ga)
        g.results.append({a: ga, b: gb})
    return g, teams


def test_tie_break_orders_by_points():
    g, _ = _group_with([("A", "B", 1, 0), ("C", "D", 0, 0), ("A", "C", 2, 0),
                        ("D", "B", 1, 0), ("D", "A", 0, 1), ("B", "C", 0, 1)])
    order = [t.name for t in g.tie_break([g.s1, g.s2, g.s3, g.s4])]
    assert order[0] == "A"
    assert order[-1] == "B"


def test_tie_break_uses_head_to_head_before_goal_difference():
    # B y C empatan a puntos; C tiene mejor DG global, pero B ganó el directo.
    g, _ = _group_with([("B", "C", 1, 0), ("A", "B", 0, 0), ("A", "C", 0, 0),
                        ("D", "B", 0, 0), ("C", "D", 5, 0), ("A", "D", 1, 0)])
    names = [t.name for t in g.tie_break([g.s2, g.s3])]
    assert names == ["B", "C"]
