"""
Fetch completed FIFA World Cup 2026 match results from football-data.org
and store them in data/world_cup_results.csv.

Usage:
    export FOOTBALL_DATA_API_KEY=your_key_here
    python3 scripts/fetch_real_results.py

The API key must be set as an environment variable. Get a free key at:
https://www.football-data.org/client/register

Runs idempotently: overwrites the output file on each call.
"""
import csv
import json
import os
import sys
import urllib.request
import urllib.error

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(ROOT, "data", "world_cup_results.csv")

API_BASE = "https://api.football-data.org/v4"
# Competition code for FIFA World Cup (used for all editions on football-data.org)
COMPETITION = "WC"

# Map football-data.org team names → names used in our model CSVs
NAME_MAP = {
    "Korea Republic":          "South Korea",
    "Republic of Korea":       "South Korea",
    "DR Congo":                "DR Congo",
    "Congo DR":                "DR Congo",
    "Democratic Republic of Congo": "DR Congo",
    "Côte d'Ivoire":           "Ivory Coast",
    "Cote d'Ivoire":           "Ivory Coast",
    "Ivory Coast":             "Ivory Coast",
    "Bosnia & Herzegovina":    "Bosnia and Herzegovina",
    "Bosnia-Herzegovina":      "Bosnia and Herzegovina",
    "Czechia":                 "Czech Republic",
    "Czech Republic":          "Czech Republic",
    "USA":                     "United States",
    "United States":           "United States",
    "Curacao":                 "Curaçao",
    "Curaçao":                 "Curaçao",
    "Cape Verde Islands":      "Cape Verde",
    "Cape Verde":              "Cape Verde",
}


def normalize(name: str) -> str:
    return NAME_MAP.get(name, name)


def fetch_json(url: str, api_key: str) -> dict:
    req = urllib.request.Request(url, headers={"X-Auth-Token": api_key})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read())


def fetch_results(api_key: str) -> list[dict]:
    url = f"{API_BASE}/competitions/{COMPETITION}/matches?status=FINISHED"
    try:
        data = fetch_json(url, api_key)
    except urllib.error.HTTPError as e:
        print(f"  HTTP {e.code}: {e.reason}", file=sys.stderr)
        if e.code == 404:
            print("  WC 2026 competition not found — may not be available in this API tier yet.",
                  file=sys.stderr)
        raise

    matches = data.get("matches", [])
    print(f"  Fetched {len(matches)} finished matches from football-data.org")

    results = []
    for m in matches:
        home = normalize(m["homeTeam"]["name"])
        away = normalize(m["awayTeam"]["name"])
        score = m.get("score", {})
        full  = score.get("fullTime", {})
        home_goals = full.get("home")
        away_goals = full.get("away")

        if home_goals is None or away_goals is None:
            continue  # score not yet recorded

        utc_date = m.get("utcDate", "")[:10]  # keep only YYYY-MM-DD
        results.append({
            "date":       utc_date,
            "home_team":  home,
            "away_team":  away,
            "home_goals": int(home_goals),
            "away_goals": int(away_goals),
        })

    results.sort(key=lambda r: r["date"])
    return results


def write_csv(results: list[dict]) -> None:
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "home_team", "away_team", "home_goals", "away_goals"])
        writer.writeheader()
        writer.writerows(results)
    print(f"  Wrote {len(results)} results → {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    api_key = os.environ.get("FOOTBALL_DATA_API_KEY", "").strip()
    if not api_key:
        print("ERROR: FOOTBALL_DATA_API_KEY env var is not set.", file=sys.stderr)
        print("Get a free key at https://www.football-data.org/client/register", file=sys.stderr)
        sys.exit(1)

    print("Fetching World Cup 2026 results...")
    results = fetch_results(api_key)
    if not results:
        print("  No finished matches yet — output file not updated.")
        sys.exit(0)
    write_csv(results)
    print("Done.")
