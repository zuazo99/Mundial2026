"""Construye el histórico de La Liga (1993/94 →) desde el espejo datasets/football-datasets + openfootball.

    python scripts/build_liga_history.py [--out data/liga/raw_hist] [--accept-changes]

* El espejo se descarga FIJADO a un commit (data/liga/source_pins.json; si no hay, se toma el HEAD con
  `git ls-remote` y se registra) y cada fichero se verifica contra su SHA-256; si cambia, falla.
* openfootball (ya limpio, en data/liga/raw) manda donde existe; el espejo aporta el histórico anterior y
  las filas de fuente única que superan la cuarentena.
* No toca data/liga/raw: escribe en --out (SP1 fusionado + copia de SP2 para sembrar ascendidos) y un informe.
"""
import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from config import path  # noqa: E402
from liga import ingest, sources  # noqa: E402

PINS = path("data", "liga", "source_pins.json")
REPORT = path("data", "liga", "history_report.json")


def season_codes(first_year: int, today: datetime.date) -> list:
    y_end = today.year if today.month >= 8 else today.year - 1
    return [f"{y % 100:02d}{(y + 1) % 100:02d}" for y in range(first_year, y_end + 1)]


def resolve_commit(pins: dict) -> str:
    if pins.get("commit"):
        return pins["commit"]
    out = subprocess.run(["git", "ls-remote", f"https://github.com/{sources.MIRROR_REPO}", "HEAD"],
                         capture_output=True, text=True, timeout=60, check=True).stdout
    return out.split()[0]


def main(out_dir: str, accept_changes: bool, fetch=sources.http_fetch, today=None) -> dict:
    today = today or datetime.date.today()
    pins = json.load(open(PINS, encoding="utf-8")) if os.path.exists(PINS) else {}
    pins.setdefault("mirror", {"repo": sources.MIRROR_REPO, "commit": None, "sha256": {}})
    commit = resolve_commit(pins["mirror"])
    bodies, missing = {}, []
    for code in season_codes(1993, today):
        try:
            bodies[code] = fetch(sources.mirror_url(code, commit))
        except sources.NotFound:
            missing.append(code)
    gaps = [c for c in missing if ingest.season_key(c) < ingest.season_key(max(bodies, key=ingest.season_key))]
    if gaps:
        raise SystemExit(f"Faltan temporadas intermedias del espejo: {gaps}")
    ref = ingest.load_seasons(path("data", "liga", "raw"))
    seasons, report = sources.build_history(bodies, ref, today, pins["mirror"]["sha256"], accept_changes)
    report.update(mirror_commit=commit, mirror_missing_trailing=missing, generated_for=today.isoformat())

    os.makedirs(out_dir, exist_ok=True)
    for code, df in seasons.items():
        sources.to_footballdata_frame(df, "SP1").to_csv(os.path.join(out_dir, f"SP1_{code}.csv"), index=False)
    for f in os.listdir(path("data", "liga", "raw")):
        if f.startswith("SP2_"):
            shutil.copy(os.path.join(path("data", "liga", "raw"), f), out_dir)
    pins["mirror"].update(commit=commit, sha256=report["mirror_sha256"])
    json.dump(pins, open(PINS, "w", encoding="utf-8"), indent=1)
    json.dump(report, open(REPORT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"✅ {len(seasons)} temporadas → {out_dir} (espejo @ {commit[:10]}); informe: {REPORT}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=path("data", "liga", "raw_hist"))
    ap.add_argument("--accept-changes", action="store_true")
    a = ap.parse_args()
    main(a.out, a.accept_changes)
