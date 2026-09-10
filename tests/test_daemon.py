from types import SimpleNamespace

from powernap.config import Config
from powernap.daemon import Daemon
from powernap.model import Decision, Profile, ThermalState


def bare_daemon(policy):
    daemon = object.__new__(Daemon)
    daemon.cfg = Config(external_change_policy=policy)
    daemon.yielded = False
    daemon.controller = SimpleNamespace(
        expected_state={"x": "expected"},
        external_changes=lambda: {"x": ("expected", "actual")},
        snapshot=lambda: {"x": "actual"},
    )
    return daemon


def test_external_observe_accepts_new_state():
    daemon = bare_daemon("observe")
    assert daemon._resolve_external_changes() is False
    assert daemon.controller.expected_state == {"x": "actual"}


def test_external_yield_stops_management():
    daemon = bare_daemon("yield")
    assert daemon._resolve_external_changes() is False
    assert daemon.yielded


def test_external_manage_requests_reapply():
    assert bare_daemon("manage")._resolve_external_changes() is True


def test_protected_workload_cannot_exceed_thermal_ceiling(monkeypatch):
    daemon = object.__new__(Daemon)
    daemon.cfg = Config(protected_processes=("work",))
    monkeypatch.setattr("powernap.daemon.protected_active", lambda names: (True, ("work",)))
    decision = Decision(0, 0, Profile.ECO, Profile.ECO, Profile.BALANCED, Profile.ECO, ThermalState.HOT, "idle")
    protected = daemon._protect_workload(decision)
    assert protected.demand_floor == Profile.RESPONSIVE
    assert protected.recommended == Profile.BALANCED
