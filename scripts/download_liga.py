"""Descarga los CSV de La Liga (SP1) y Segunda (SP2) de football-data.co.uk a data/liga/raw/.

Uso (en local, con red):
    python scripts/download_liga.py              # últimas 5 temporadas
    python scripts/download_liga.py --seasons 8  # más historia
    python scripts/download_liga.py --refresh    # vuelve a bajar también las ya existentes

Es idempotente: las temporadas ya terminadas que existen en disco no se
vuelven a descargar; la temporada en curso se refresca siempre.
"""
import argparse
import datetime
import os
import sys
import tempfile
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from config import path  # noqa: E402

URL = "https://www.football-data.co.uk/mmz4281/{code}/{division}.csv"
RAW_DIR = path("data", "liga", "raw")


def current_season_start_year(today: datetime.date) -> int:
    """La temporada empieza en agosto: nov-2026 → 2026; mar-2027 → 2026."""
    return today.year if today.month >= 8 else today.year - 1


def season_codes(today: datetime.date, n: int) -> list:
    """Códigos 'aabb' de las últimas n temporadas, de más antigua a actual."""
    y = current_season_start_year(today)
    return [f"{(y - k) % 100:02d}{(y - k + 1) % 100:02d}" for k in range(n - 1, -1, -1)]


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "mundial2026-liga-ingest"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def download(codes: list, raw_dir: str = RAW_DIR, refresh: bool = False, fetcher=fetch,
             division: str = "SP1") -> list:
    """Descarga las temporadas que falten. Devuelve los códigos descargados."""
    os.makedirs(raw_dir, exist_ok=True)
    current = codes[-1]
    done = []
    for code in codes:
        dest = os.path.join(raw_dir, f"{division}_{code}.csv")
        if os.path.exists(dest) and not refresh and code != current:
            print(f"  = {division}_{code}.csv ya existe")
            continue
        data = fetcher(URL.format(code=code, division=division))
        if b"HomeTeam" not in data[:2000]:
            raise RuntimeError(f"{code}: la respuesta no parece un CSV de football-data")
        fd, tmp = tempfile.mkstemp(dir=raw_dir)
        with os.fdopen(fd, "wb") as f:  # escritura atómica
            f.write(data)
        os.replace(tmp, dest)
        print(f"  ✅ {division}_{code}.csv ({len(data)//1024} KB)")
        done.append(code)
    return done


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", type=int, default=5)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--no-segunda", action="store_true", help="no bajar SP2 (siembra de ascendidos)")
    a = ap.parse_args()
    codes = season_codes(datetime.date.today(), a.seasons)
    download(codes, refresh=a.refresh)
    if not a.no_segunda:
        download(codes, refresh=a.refresh, division="SP2")
