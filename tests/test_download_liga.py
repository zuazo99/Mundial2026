"""Descargador de La Liga: lógica de temporadas e idempotencia (sin red)."""
import datetime
import importlib.util
import os

import pytest

_spec = importlib.util.spec_from_file_location(
    "download_liga", os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts", "download_liga.py"))
dl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dl)

CSV = b"Div,Date,HomeTeam,AwayTeam,FTHG,FTAG\nSP1,01/01/22,A,B,1,0\n"


def test_season_codes_switch_in_august():
    assert dl.season_codes(datetime.date(2026, 11, 5), 5) == ["2223", "2324", "2425", "2526", "2627"]
    assert dl.season_codes(datetime.date(2027, 3, 1), 2) == ["2526", "2627"]
    assert dl.season_codes(datetime.date(2027, 8, 1), 1) == ["2728"]
    assert dl.season_codes(datetime.date(2099, 9, 1), 2) == ["9899", "9900"]  # cruce de siglo


def test_download_skips_finished_seasons_but_refreshes_current(tmp_path):
    calls = []
    fetcher = lambda url: calls.append(url) or CSV
    assert dl.download(["2425", "2526"], str(tmp_path), fetcher=fetcher) == ["2425", "2526"]
    calls.clear()
    assert dl.download(["2425", "2526"], str(tmp_path), fetcher=fetcher) == ["2526"]  # solo la actual
    assert len(calls) == 1 and "2526" in calls[0]
    calls.clear()
    assert dl.download(["2425", "2526"], str(tmp_path), refresh=True, fetcher=fetcher) == ["2425", "2526"]


def test_download_rejects_non_csv_response_and_leaves_no_file(tmp_path):
    with pytest.raises(RuntimeError):
        dl.download(["2526"], str(tmp_path), fetcher=lambda u: b"<html>blocked</html>")
    assert os.listdir(tmp_path) == []
