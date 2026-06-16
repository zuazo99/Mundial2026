"""
Monte Carlo tournament simulator.

Runs N full tournament simulations per model variant using:
  - Real XGBoost model (from api/models/) for knockout XG
  - J1/J2/J3 XG predictions for group stage
  - Single Poisson draw per match (no modal averaging)

Outputs: results/probabilities_<variant>.json
  { "champion": { "Spain": 0.23, ... }, "finalist": {...}, "semis": {...}, "n_sims": 1000 }

Usage (from repo root):
    python3 src/monte_carlo.py [n_sims]   # default 1000
"""

import csv
import json
import os
import sys
import numpy as np
import xgboost as xgb
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VARIANTS = ["misterclaude", "gemaldini", "dav_gpo"]

# Group structure — mirrors groups.ts and clases_simulacion.py
GROUPS = [
    ("A", "Mexico",        "South Africa",          "South Korea",   "Czech Republic"      ),
    ("B", "Canada",        "Bosnia and Herzegovina", "Qatar",         "Switzerland"         ),
    ("C", "Brazil",        "Morocco",               "Haiti",         "Scotland"            ),
    ("D", "United States", "Paraguay",              "Australia",     "Turkey"              ),
    ("E", "Germany",       "Curaçao",               "Ivory Coast",   "Ecuador"             ),
    ("F", "Netherlands",   "Japan",                 "Sweden",        "Tunisia"             ),
    ("G", "Belgium",       "Egypt",                 "Iran",          "New Zealand"         ),
    ("H", "Spain",         "Cape Verde",            "Saudi Arabia",  "Uruguay"             ),
    ("I", "France",        "Senegal",               "Iraq",          "Norway"              ),
    ("J", "Argentina",     "Algeria",               "Austria",       "Jordan"              ),
    ("K", "Portugal",      "DR Congo",              "Uzbekistan",    "Colombia"            ),
    ("L", "England",       "Croatia",               "Ghana",         "Panama"              ),
]

# (slot_i, slot_j) pairs for each match in a group
# Slots are 0-indexed (s1=0, s2=1, s3=2, s4=3)
GROUP_FIXTURES = [
    (0, 1), (2, 3),   # J1
    (0, 2), (3, 1),   # J2
    (3, 0), (1, 2),   # J3
]

FEATURES = [
    'elo', 'opponent_elo', 'is_home', 'tournament_num', 'confed', 'rival_confed',
    'gf_prom_5', 'gc_prom_5', 'elo_prom_5', 'gf_prom_15', 'gc_prom_15', 'PCA_1', 'PCA_2',
    'rival_gf_prom_5', 'rival_gc_prom_5', 'rival_elo_prom_5', 'rival_gf_prom_15',
    'rival_gc_prom_15', 'rival_PCA_1', 'rival_PCA_2',
]


# ─── helpers ─────────────────────────────────────────────────────────────────

def read_csv(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def simulate_match(xg1: float, xg2: float, ko: bool = False):
    """Single Poisson draw. KO adds extra time + penalties on draw."""
    g1 = int(np.random.poisson(xg1))
    g2 = int(np.random.poisson(xg2))
    if ko and g1 == g2:
        g1 += int(np.random.poisson(xg1 / 3))
        g2 += int(np.random.poisson(xg2 / 3))
        if g1 == g2:
            p1 = sum(1 for _ in range(5) if np.random.random() < 0.75)
            p2 = sum(1 for _ in range(5) if np.random.random() < 0.75)
            while p1 == p2:
                p1 += int(np.random.random() < 0.75)
                p2 += int(np.random.random() < 0.75)
            if p1 > p2:
                g1 += 1
            else:
                g2 += 1
    return g1, g2


def sort_group(teams_stats: list[dict]) -> list[dict]:
    """Sort by pts → gd → gf (simplified tiebreak for Monte Carlo speed)."""
    return sorted(teams_stats, key=lambda t: (t['pts'], t['gd'], t['gf']), reverse=True)


# ─── data loaders ────────────────────────────────────────────────────────────

def load_group_xg(variant: str) -> dict:
    """xg[team][opponent] = predicted XG, from J1/J2/J3 CSVs."""
    xg: dict = {}
    for j in ['J1', 'J2', 'J3']:
        for row in read_csv(os.path.join(ROOT, "data", "ai_models", f"xg_preds_{j}_{variant}.csv")):
            t, opp = row['team'], row['opponent']
            xg.setdefault(t, {})[opp] = float(row['xg_estimated'])
    return xg


def load_knockout_xg(variant: str) -> dict:
    """
    Predict XG for all 48×47 team pairs using the saved XGBoost model
    and the foto_fija snapshot (last known state per team before the WC).
    Returns nested dict: xg[t1][t2] = xg of t1 vs t2.
    """
    model = xgb.XGBRegressor()
    model.load_model(os.path.join(ROOT, "api", "models", f"xgb_{variant}.json"))

    rows = read_csv(os.path.join(ROOT, "data", "ai_models", f"xg_preds_J1_{variant}_complete.csv"))
    teams: dict = {}
    for r in rows:
        if r['team'] not in teams:
            teams[r['team']] = r

    team_names = sorted(teams.keys())
    feat_rows, pair_index = [], []
    for t1 in team_names:
        for t2 in team_names:
            if t1 == t2:
                continue
            s1, s2 = teams[t1], teams[t2]
            feat_rows.append([
                float(s1['elo']), float(s2['elo']), 0, 5,
                int(s1['confed']), int(s2['confed']),
                float(s1['gf_prom_5']), float(s1['gc_prom_5']), float(s1['elo_prom_5']),
                float(s1['gf_prom_15']), float(s1['gc_prom_15']),
                float(s1['PCA_1']), float(s1['PCA_2']),
                float(s2['gf_prom_5']), float(s2['gc_prom_5']), float(s2['elo_prom_5']),
                float(s2['gf_prom_15']), float(s2['gc_prom_15']),
                float(s2['PCA_1']), float(s2['PCA_2']),
            ])
            pair_index.append((t1, t2))

    preds = model.predict(np.array(feat_rows, dtype=np.float32))
    nested: dict = {}
    for (t1, t2), xg_val in zip(pair_index, preds):
        nested.setdefault(t1, {})[t2] = float(xg_val)
    return nested


def load_mejores_terceros() -> dict:
    path = os.path.join(ROOT, "data", "mejores_terceros.csv")
    with open(path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=";")
        return {r["Combinación"]: {
            "1A": r["1A"], "1B": r["1B"], "1D": r["1D"], "1E": r["1E"],
            "1G": r["1G"], "1I": r["1I"], "1K": r["1K"], "1L": r["1L"],
        } for r in reader}


# ─── simulation ──────────────────────────────────────────────────────────────

def simulate_group_stage(group_xg: dict) -> list[dict]:
    """Returns list of {letter, ranked: [{team, pts, gd, gf}, ...]}."""
    results = []
    for (letter, *slots) in GROUPS:
        stats = {t: {'pts': 0, 'gd': 0, 'gf': 0} for t in slots}
        for (i, j) in GROUP_FIXTURES:
            t1, t2 = slots[i], slots[j]
            xg1 = group_xg.get(t1, {}).get(t2, 1.2)
            xg2 = group_xg.get(t2, {}).get(t1, 1.2)
            g1, g2 = simulate_match(xg1, xg2, ko=False)
            stats[t1]['gf'] += g1
            stats[t2]['gf'] += g2
            stats[t1]['gd'] += g1 - g2
            stats[t2]['gd'] += g2 - g1
            if g1 > g2:
                stats[t1]['pts'] += 3
            elif g1 < g2:
                stats[t2]['pts'] += 3
            else:
                stats[t1]['pts'] += 1
                stats[t2]['pts'] += 1
        ranked = sort_group([{'team': t, **s} for t, s in stats.items()])
        results.append({'letter': letter, 'ranked': ranked})
    return results


def build_r32(group_results: list[dict], mejores_terceros: dict) -> list[tuple]:
    """Assemble R32 bracket — mirrors clases_simulacion.py's create_first_round()."""
    W = {g['letter']: g['ranked'][0]['team'] for g in group_results}
    R = {g['letter']: g['ranked'][1]['team'] for g in group_results}

    # Best 8 thirds by pts → gd → gf
    thirds_all = sorted(
        [{'team': g['ranked'][2]['team'], 'letter': g['letter'], **g['ranked'][2]}
         for g in group_results],
        key=lambda t: (t['pts'], t['gd'], t['gf']), reverse=True
    )[:8]

    letters = ''.join(sorted(t['letter'] for t in thirds_all))
    lookup = mejores_terceros.get(letters, {})
    third_map: dict = {}
    for slot, group_ref in lookup.items():
        letter = group_ref[1:]  # "3E" → "E"
        team = next((t['team'] for t in thirds_all if t['letter'] == letter), None)
        if team:
            third_map[slot] = team

    if not third_map:
        for i, slot in enumerate(["1A", "1B", "1D", "1E", "1G", "1I", "1K", "1L"]):
            if i < len(thirds_all):
                third_map[slot] = thirds_all[i]['team']

    def tm(slot):
        return third_map.get(slot, thirds_all[0]['team'] if thirds_all else "TBD")

    return [
        (R['A'], R['B']),      (W['C'], R['F']),
        (W['E'], tm('1E')),    (W['F'], R['C']),
        (R['E'], R['I']),      (W['I'], tm('1I')),
        (W['A'], tm('1A')),    (W['L'], tm('1L')),
        (W['G'], tm('1G')),    (W['D'], tm('1D')),
        (W['H'], R['J']),      (R['K'], R['L']),
        (W['B'], tm('1B')),    (R['D'], R['G']),
        (W['J'], R['H']),      (W['K'], tm('1K')),
    ]


def play_ko_round(fixtures: list[tuple], ko_xg: dict) -> list[str]:
    winners = []
    for t1, t2 in fixtures:
        xg1 = ko_xg.get(t1, {}).get(t2, 1.2)
        xg2 = ko_xg.get(t2, {}).get(t1, 1.2)
        g1, g2 = simulate_match(xg1, xg2, ko=True)
        winners.append(t1 if g1 > g2 else t2)
    return winners


def run_tournament(group_xg: dict, ko_xg: dict, mejores_terceros: dict) -> dict:
    """Run one full WC simulation. Returns round-reach info per team."""
    reached: dict[str, set] = {}

    def mark(teams, rnd):
        for t in teams:
            reached.setdefault(t, set()).add(rnd)

    group_results = simulate_group_stage(group_xg)

    r32 = build_r32(group_results, mejores_terceros)
    r32_all = [t for pair in r32 for t in pair]
    mark(r32_all, 'r32')

    r32_w = play_ko_round(r32, ko_xg)
    mark(r32_w, 's16')

    s16 = [
        (r32_w[0], r32_w[3]),  (r32_w[2], r32_w[5]),
        (r32_w[1], r32_w[4]),  (r32_w[6], r32_w[7]),
        (r32_w[10], r32_w[11]),(r32_w[9], r32_w[8]),
        (r32_w[14], r32_w[13]),(r32_w[12], r32_w[15]),
    ]
    s16_w = play_ko_round(s16, ko_xg)
    mark(s16_w, 'e8')

    e8 = [
        (s16_w[0], s16_w[1]), (s16_w[4], s16_w[5]),
        (s16_w[2], s16_w[3]), (s16_w[6], s16_w[7]),
    ]
    e8_w = play_ko_round(e8, ko_xg)
    mark(e8_w, 'semis')

    semi_w = play_ko_round([(e8_w[0], e8_w[1]), (e8_w[2], e8_w[3])], ko_xg)
    mark(semi_w, 'final')

    finalists = [semi_w[0], semi_w[1]]
    t1, t2 = finalists
    xg1 = ko_xg.get(t1, {}).get(t2, 1.2)
    xg2 = ko_xg.get(t2, {}).get(t1, 1.2)
    g1, g2 = simulate_match(xg1, xg2, ko=True)
    champion = t1 if g1 > g2 else t2
    mark([champion], 'champion')

    return reached


# ─── main ────────────────────────────────────────────────────────────────────

def run_monte_carlo(variant: str, n_sims: int) -> dict:
    print(f"\n[{variant}] Cargando datos...")
    group_xg     = load_group_xg(variant)
    ko_xg        = load_knockout_xg(variant)
    mejores_terc = load_mejores_terceros()

    all_teams = [t for (_, *slots) in GROUPS for t in slots]
    counts: dict[str, dict[str, int]] = {t: defaultdict(int) for t in all_teams}

    print(f"[{variant}] Simulando {n_sims} torneos...")
    for i in range(n_sims):
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{n_sims}…")
        reached = run_tournament(group_xg, ko_xg, mejores_terc)
        for team, rounds in reached.items():
            for rnd in rounds:
                counts.setdefault(team, defaultdict(int))[rnd] += 1

    def prob(team, rnd):
        return round(counts.get(team, {}).get(rnd, 0) / n_sims, 4)

    result = {
        'variant': variant,
        'n_sims':  n_sims,
        'teams': {
            t: {
                'pR32':     prob(t, 'r32'),
                'pS16':     prob(t, 's16'),
                'pE8':      prob(t, 'e8'),
                'pSemis':   prob(t, 'semis'),
                'pFinal':   prob(t, 'final'),
                'pChampion': prob(t, 'champion'),
            }
            for t in all_teams
        },
    }

    # Sort teams by P(champion) desc for convenience
    result['teams'] = dict(
        sorted(result['teams'].items(), key=lambda x: x[1]['pChampion'], reverse=True)
    )
    return result


if __name__ == '__main__':
    n_sims = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    os.makedirs(os.path.join(ROOT, 'results'), exist_ok=True)

    for variant in VARIANTS:
        data = run_monte_carlo(variant, n_sims)
        out = os.path.join(ROOT, 'results', f'probabilities_{variant}.json')
        with open(out, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        top3 = list(data['teams'].items())[:3]
        print(f"  Top 3: " + ", ".join(f"{t} {v['pChampion']*100:.1f}%" for t, v in top3))
        print(f"  ✅ {out}")

    print("\n✅ Monte Carlo completo.")
