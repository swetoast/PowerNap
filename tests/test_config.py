from pathlib import Path
import pytest
from powernap.config import ConfigError, load_config


def test_boolean_with_inline_comment(tmp_path: Path):
    path = tmp_path / "powernap.conf"
    path.write_text("[cpu]\nenabled = true # comment\n")
    assert load_config(path).manage_cpu is True


def test_invalid_temperature_order(tmp_path: Path):
    path = tmp_path / "powernap.conf"
    path.write_text("[thermal]\nwarm_temp_c=80\nhot_temp_c=70\ncritical_temp_c=90\n")
    with pytest.raises(ConfigError): load_config(path)


def test_explicit_missing_config_is_error(tmp_path: Path):
    with pytest.raises(ConfigError, match="does not exist"):
        load_config(tmp_path / "missing.conf")


def test_invalid_ranges_are_rejected(tmp_path: Path):
    path = tmp_path / "powernap.conf"
    path.write_text("[price]\nlookahead_hours=0\n[sampling]\nbusy_core_threshold=101\n")
    with pytest.raises(ConfigError):
        load_config(path)


def test_unknown_price_provider_is_rejected(tmp_path: Path):
    path = tmp_path / "powernap.conf"
    path.write_text("[price]\nprovider=unknown\n")
    with pytest.raises(ConfigError, match="provider is unsupported"):
        load_config(path)
