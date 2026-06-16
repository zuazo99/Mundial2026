"""
Update each variant's "foto fija" (team stat snapshot) using real World Cup results.

Reads:  data/world_cup_results.csv
        data/models_csv/df_{variant}.csv   (historical match data per team)
Writes: data/ai_models/foto_fija_updated.csv  (variant-independent since real scores are facts)

The updated file is picked up automatically by src/xg_preds.py when it exists.
Run AFTER fetch_real_results.py.

Usage:
    python3 scripts/update_foto_fija.py
"""
import csv
import os
import math

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REAL_RESULTS = os.path.join(ROOT, "data", "world_cup_results.csv")
OUT = os.path.join(ROOT, "data", "ai_models", "foto_fija_updated.csv")

VARIANTS = ["misterclaude", "gemaldini", "dav_gpo"]

# FIFA World Cup ELO K-factor
ELO_K = 40


def read_csv(path: str) -> list[dict]:
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def elo_expected(team_elo: float, opp_elo: float) -> float:
    return 1.0 / (1.0 + 10 ** ((opp_elo - team_elo) / 400.0))


def elo_update(team_elo: float, opp_elo: float, result: float) -> float:
    """result: 1=win, 0.5=draw, 0=loss"""
    return team_elo + ELO_K * (result - elo_expected(team_elo, opp_elo))


def rolling_mean(values: list[float], window: int) -> float:
    tail = values[-window:] if len(values) >= window else values
    return sum(tail) / len(tail) if tail else 0.0


def load_foto_fija(variant: str) -> dict[str, dict]:
    """Load pre-WC foto fija from the J1 complete CSV (one row = one predicted match)."""
    path = os.path.join(ROOT, "data", "ai_models", f"xg_preds_J1_{variant}_complete.csv")
    rows = read_csv(path)
    teams: dict[str, dict] = {}
    for r in rows:
        name = r["team"]
        if name in teams:
            continue
        try:
            teams[name] = {
                "elo":      float(r["elo"]),
                "gf_prom_5":  float(r["gf_prom_5"]),
                "gc_prom_5":  float(r["gc_prom_5"]),
                "elo_prom_5": float(r["elo_prom_5"]),
                "gf_prom_15": float(r["gf_prom_15"]),
                "gc_prom_15": float(r["gc_prom_15"]),
                "PCA_1":    float(r["PCA_1"]),
                "PCA_2":    float(r["PCA_2"]),
                "confed":   int(r["confed"]),
            }
        except (KeyError, ValueError):
            pass
    return teams


def load_recent_history(variant: str, window: int = 15) -> dict[str, list[dict]]:
    """
    Load the last `window` historical matches per team from the training CSV.
    Returns {team_name: [{"goals": int, "goals_conceded": int, "opponent_elo": float}, ...]}
    sorted oldest→newest.
    """
    path = os.path.join(ROOT, "data", "models_csv", f"df_{variant}.csv")
    rows = read_csv(path)
    # keep only pre-WC rows where goals are recorded
    history: dict[str, list] = {}
    for r in rows:
        if r.get("date", "") >= "2026-06-11":
            continue
        goals = r.get("goals", "")
        goals_c = r.get("goals_conceded", "")
        opp_elo = r.get("opponent_elo", "")
        if goals == "" or goals_c == "" or opp_elo == "":
            continue
        team = r["team"]
        if team not in history:
            history[team] = []
        history[team].append({
            "date":           r["date"],
            "goals":          float(goals),
            "goals_conceded": float(goals_c),
            "opponent_elo":   float(opp_elo),
        })
    # sort by date and keep last `window` entries
    for team in history:
        history[team] = sorted(history[team], key=lambda x: x["date"])[-window:]
    return history


def compute_updated_foto_fija(foto_fija: dict, history: dict, real_results: list[dict]) -> dict:
    """
    Apply real WC match results on top of the foto fija.
    Returns updated foto fija dict (same structure as input `foto_fija`).
    """
    # Clone the foto fija
    updated = {team: dict(stats) for team, stats in foto_fija.items()}
    # Clone history buffers (used to compute new rolling avgs)
    buffers: dict[str, list] = {}
    for team, matches in history.items():
        buffers[team] = [dict(m) for m in matches]

    # Process real matches in date order
    for match in sorted(real_results, key=lambda x: x["date"]):
        home = match["home_team"]
        away = match["away_team"]
        hg   = int(match["home_goals"])
        ag   = int(match["away_goals"])

        for team, opp, gf, gc in [(home, away, hg, ag), (away, home, ag, hg)]:
            if team not in updated:
                continue  # team not in our model (shouldn't happen)

            opp_stats = updated.get(opp, {})
            opp_elo   = opp_stats.get("elo", updated[team]["elo"])

            # ELO update
            if gf > gc:
                result = 1.0
            elif gf == gc:
                result = 0.5
            else:
                result = 0.0
            updated[team]["elo"] = elo_update(updated[team]["elo"], opp_elo, result)

            # Extend rolling buffer
            buf = buffers.setdefault(team, [])
            buf.append({
                "date":           match["date"],
                "goals":          float(gf),
                "goals_conceded": float(gc),
                "opponent_elo":   opp_elo,
            })

            # Recompute rolling averages from buffer
            gf_list  = [x["goals"]          for x in buf]
            gc_list  = [x["goals_conceded"] for x in buf]
            oelo_list= [x["opponent_elo"]    for x in buf]

            updated[team]["gf_prom_5"]  = rolling_mean(gf_list,   5)
            updated[team]["gc_prom_5"]  = rolling_mean(gc_list,   5)
            updated[team]["elo_prom_5"] = rolling_mean(oelo_list, 5)
            updated[team]["gf_prom_15"] = rolling_mean(gf_list,  15)
            updated[team]["gc_prom_15"] = rolling_mean(gc_list,  15)

    return updated


def write_foto_fija(updated: dict) -> None:
    fieldnames = ["team", "elo", "gf_prom_5", "gc_prom_5", "elo_prom_5",
                  "gf_prom_15", "gc_prom_15", "PCA_1", "PCA_2", "confed"]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for team, stats in sorted(updated.items()):
            writer.writerow({"team": team, **{k: stats[k] for k in fieldnames[1:] if k in stats}})
    print(f"  Wrote updated foto fija for {len(updated)} teams → {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    if not os.path.exists(REAL_RESULTS):
        print(f"No real results file found at {os.path.relpath(REAL_RESULTS, ROOT)}.")
        print("Run scripts/fetch_real_results.py first.")
        raise SystemExit(1)

    real_results = read_csv(REAL_RESULTS)
    if not real_results:
        print("No real results yet — nothing to update.")
        raise SystemExit(0)

    print(f"Processing {len(real_results)} real matches...")

    # Use misterclaude as canonical foto fija base (all variants share the same WC calendar)
    # ELO/rolling-avg updates are applied identically since real results are factual
    foto_fija = load_foto_fija("misterclaude")
    history   = load_recent_history("misterclaude")

    updated = compute_updated_foto_fija(foto_fija, history, real_results)

    teams_updated = sum(
        1 for team in updated
        if any(
            m["home_team"] == team or m["away_team"] == team
            for m in real_results
        )
    )
    print(f"  ELO + rolling averages updated for {teams_updated} teams")
    write_foto_fija(updated)
    print("Done.")
