"""config.py: rutas absolutas y constantes coherentes."""
import os

import config


def test_path_is_absolute_and_independent_of_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = config.path("data", "mejores_terceros.csv")
    assert os.path.isabs(p) and os.path.exists(p)


def test_wc_dates_are_ordered():
    assert config.WC_START < config.WC_J2_START < config.WC_J3_START


def test_every_variant_has_saved_model_and_rho():
    for v in config.VARIANTS:
        assert os.path.exists(config.path("api", "models", f"xgb_{v}.json"))
        assert os.path.exists(config.path("api", "models", f"rho_{v}.json"))
