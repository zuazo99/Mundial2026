"""Descarga y limpia La Liga desde fuentes abiertas (openfootball/football.json, CC0).

    python scripts/fetch_liga_openfootball.py            # últimas 11 temporadas (hasta la actual)
    python scripts/fetch_liga_openfootball.py --seasons 6

Escribe data/liga/raw/SP1_<aabb>.csv y SP2_<aabb>.csv (formato football-data, ya limpios),
data/liga/cleaning_report.json y una caché del JSON original en data/liga/source/ (en .gitignore).
Después: python src/liga/ingest.py && python src/liga/model.py && python src/liga/simulate.py
"""
import argparse
import datetime
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from config import path  # noqa: E402
from liga import sources  # noqa: E402


def season_labels(today: datetime.date, n: int) -> list:
    """['2016-17', ..., '2026-27']: la temporada empieza en agosto."""
    y = today.year if today.month >= 8 else today.year - 1
    return [f"{y - k}-{str(y - k + 1)[2:]}" for k in range(n - 1, -1, -1)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", type=int, default=11)
    a = ap.parse_args()
    today = datetime.date.today()
    rep = sources.build(season_labels(today, a.seasons), today, sources.http_fetch,
                        raw_dir=path("data", "liga", "raw"), cache_dir=path("data", "liga", "source"),
                        report_path=path("data", "liga", "cleaning_report.json"))
    print(f"✅ datos a fecha {rep['data_as_of']}; informe en data/liga/cleaning_report.json")
