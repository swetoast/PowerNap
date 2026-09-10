import json
from pathlib import Path
from unittest.mock import patch

import pytest

from powernap.capabilities import Capabilities
from powernap.cli import main
from powernap.model import CPUState, PriceContext, SystemState


def config(tmp_path: Path, cpu=True):
    p=tmp_path/'powernap.conf'
    p.write_text(f'[general]\ndry_run=true\n[storage]\ndatabase_path={tmp_path / "x.db"}\n[cpu]\nenabled={str(cpu).lower()}\n[gpu]\nnvidia_enabled=false\namdgpu_enabled=false\n')
    return p


def test_capabilities_command_is_json(tmp_path, capsys):
    with patch('powernap.cli.discover', return_value=Capabilities()):
        assert main(['--config',str(config(tmp_path,False)),'capabilities']) == 0
    assert json.loads(capsys.readouterr().out)['cpu_policies'] == []


def test_check_fails_when_enabled_cpu_control_is_unavailable(tmp_path, capsys):
    with patch('powernap.cli.discover', return_value=Capabilities()):
        assert main(['--config',str(config(tmp_path,True)),'check']) == 1
    assert json.loads(capsys.readouterr().out)['valid'] is False


def test_check_passes_when_unavailable_adapters_are_disabled(tmp_path, capsys):
    with patch('powernap.cli.discover', return_value=Capabilities()):
        assert main(['--config',str(config(tmp_path,False)),'check']) == 0
    assert json.loads(capsys.readouterr().out)['valid'] is True


def test_once_no_price_is_dry_run_and_json(tmp_path, capsys):
    state=SystemState('now',1,CPUState(1,2,0,0,0,0,30,'cpu'),(),PriceContext())
    with patch('powernap.cli.discover', return_value=Capabilities()), patch('powernap.cli.Collector.collect', return_value=state):
        assert main(['--config',str(config(tmp_path,False)),'once','--no-price']) == 0
    payload=json.loads(capsys.readouterr().out)
    assert payload['results'] == []


def test_explicit_missing_config_exits_with_parser_error(tmp_path):
    with pytest.raises(SystemExit) as exc:
        main(['--config',str(tmp_path/'missing.conf'),'check'])
    assert exc.value.code == 2
