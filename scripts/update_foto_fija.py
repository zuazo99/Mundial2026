"""
Update the "foto fija" (team stat snapshot) used by xg_preds.py to include
real World Cup match results from data/world_cup_results.csv.

For each team that has played real WC matches this updates:
  - Rolling averages (gf_prom_5, gc_prom_5, gf_prom_15, gc_prom_15)
  - ELO (using standard FIFA-style formula: K=40)

Output: data/ai_models/foto_fija_updated.csv
        → picked up automatically by src/xg_preds.py when it exists.

Run after fetch_real_results.py:
    python3 scripts/update_foto_fija.py
"""
import csv
import os
import math

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "data", "world_cup_results.csv")
OUT     = os.path.join(ROOT, "data", "ai_models", "foto_fija_updated.csv")

VARIANTS   = ["misterclaude", "gemaldini", "dav_gpo"]
K_FACTOR   = 40   # FIFA World Cup ELO K-factor
WINDOW_5   = 5
WINDOW_15  = 15


# ─── helpers ────────────────────────────────────────────────────────────────

def read_csv(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def elo_expected(team_elo: float, opp_elo: float) -> float:
    return 1.0 / (1.0 + 10 ** ((opp_elo - team_elo) / 400.0))


def elo_update(team_elo: float, opp_elo: float, result: float) -> float:
    """result: 1=win, 0.5=draw, 0=loss"""
    return team_elo + K_FACTOR * (result - elo_expected(team_elo, opp_elo))


def rolling_avg(history: list[float], window: int) -> float:
    """Mean of up to the last `window` elements."""
    tail = history[-window:]
    return sum(tail) / len(tail) if tail else 0.0


# ─── main ───────────────────────────────────────────────────────────────────

def build_updated_foto_fija():
    if not os.path.exists(RESULTS):
        print(f"  {os.path.relpath(RESULTS, ROOT)} not found — nothing to update.")
        return

    real_matches = read_csv(RESULTS)
    if not real_matches:
        print("  No real results found — nothing to update.")
        return

    print(f"  {len(real_matches)} real matches loaded.")

    # We base the update on misterclaude's foto fija (ELO / rolling avgs are
    # the same across variants; only the trained model differs per variant).
    base_path = os.path.join(
        ROOT, "data", "ai_models", "xg_preds_J1_misterclaude_complete.csv"
    )
    base_rows = read_csv(base_path)

    # Build a per-team snapshot from the J1 complete file (last row per team)
    snapshot: dict[str, dict] = {}
    for r in base_rows:
        snapshot[r["team"]] = r

    # Per-team recent goals history (last 15 pre-WC for the rolling window)
    # We need the raw historical sequence to extend the rolling window correctly.
    # Load the training CSV for the last-match history.
    history_path = os.path.join(ROOT, "data", "models_csv", "df_misterclaude.csv")
    hist_rows = read_csv(history_path)
    # Keep only pre-WC rows, sorted by team+date
    hist_rows = [r for r in hist_rows if r["date"] < "2026-06-11"]
    hist_rows.sort(key=lambda r: (r["team"], r["date"]))

    # Build per-team goal histories (last 20 to cover window_15 + new matches)
    gf_history: dict[str, list[float]] = {}
    gc_history: dict[str, list[float]] = {}
    elo_map:    dict[str, float]        = {}

    for r in hist_rows:
        team = r["team"]
        try:
            gf = float(r["goals"]) if r["goals"] else None
            gc = float(r["goals_conceded"]) if r["goals_conceded"] else None
            el = float(r["elo"]) if r["elo"] else None
        except (ValueError, KeyError):
            continue
        if gf is not None:
            gf_history.setdefault(team, []).append(gf)
        if gc is not None:
            gc_history.setdefault(team, []).append(gc)
        if el is not None:
            elo_map[team] = el   # keep the last ELO value

    # Trim to last 20 per team so we only hold recent history
    for team in list(gf_history):
        gf_history[team] = gf_history[team][-20:]
    for team in list(gc_history):
        gc_history[team] = gc_history[team][-20:]

    # Seed any WC teams missing from history with the foto fija averages
    for team, snap in snapshot.items():
        if team not in gf_history:
            gf_history[team] = [float(snap.get("gf_prom_5") or 1.0)] * 5
        if team not in gc_history:
            gc_history[team] = [float(snap.get("gc_prom_5") or 1.0)] * 5
        if team not in elo_map:
            elo_map[team] = float(snap.get("elo") or 1500.0)

    # Apply real WC results in chronological order
    real_matches.sort(key=lambda r: r["date"])
    for m in real_matches:
        home = m["home_team"]
        away = m["away_team"]
        hg   = int(m["home_goals"])
        ag   = int(m["away_goals"])
        h_elo = elo_map.get(home, 1500.0)
        a_elo = elo_map.get(away, 1500.0)

        # ELO result: 1=win, 0.5=draw, 0=loss
        if hg > ag:
            h_res, a_res = 1.0, 0.0
        elif hg < ag:
            h_res, a_res = 0.0, 1.0
        else:
            h_res, a_res = 0.5, 0.5

        new_h_elo = elo_update(h_elo, a_elo, h_res)
        new_a_elo = elo_update(a_elo, h_elo, a_res)

        for team, gf, gc, new_elo in [
            (home, hg, ag, new_h_elo),
            (away, ag, hg, new_a_elo),
        ]:
            gf_history.setdefault(team, []).append(float(gf))
            gc_history.setdefault(team, []).append(float(gc))
            elo_map[team] = new_elo

    # Build the updated foto fija rows
    teams_with_real_results = set()
    for m in real_matches:
        teams_with_real_results.add(m["home_team"])
        teams_with_real_results.add(m["away_team"])

    updated: dict[str, dict] = {}
    for team, snap in snapshot.items():
        if team not in teams_with_real_results:
            continue   # only write rows for teams that have played
        updated[team] = {
            "team":       team,
            "elo":        round(elo_map.get(team, float(snap.get("elo") or 1500.0)), 1),
            "gf_prom_5":  round(rolling_avg(gf_history.get(team, []), WINDOW_5),  4),
            "gc_prom_5":  round(rolling_avg(gc_history.get(team, []), WINDOW_5),  4),
            "elo_prom_5": snap.get("elo_prom_5", ""),   # opponent ELO avg unchanged
            "gf_prom_15": round(rolling_avg(gf_history.get(team, []), WINDOW_15), 4),
            "gc_prom_15": round(rolling_avg(gc_history.get(team, []), WINDOW_15), 4),
            "PCA_1":      snap.get("PCA_1", ""),        # style unchanged after 1-3 matches
            "PCA_2":      snap.get("PCA_2", ""),
            "confed":     snap.get("confed", ""),
        }

    if not updated:
        print("  No teams with real results found in foto fija — nothing written.")
        return

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fields = ["team", "elo", "gf_prom_5", "gc_prom_5", "elo_prom_5",
              "gf_prom_15", "gc_prom_15", "PCA_1", "PCA_2", "confed"]
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(updated.values())

    print(f"  Updated foto fija for {len(updated)} teams → {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    print("Updating foto fija with real World Cup results...")
    build_updated_foto_fija()
    print("Done.")
