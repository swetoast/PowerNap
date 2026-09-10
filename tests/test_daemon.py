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
        yielded_targets=set(),
        yield_targets=lambda targets: daemon.controller.yielded_targets.update(targets),
    )
    return daemon


def test_external_observe_accepts_new_state():
    daemon = bare_daemon("observe")
    assert daemon._resolve_external_changes() is False
    assert daemon.controller.expected_state == {"x": "actual"}


def test_external_yield_stops_management():
    daemon = bare_daemon("yield")
    assert daemon._resolve_external_changes() is False
    assert daemon.controller.yielded_targets == {"x"}


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


def test_resume_refresh_preserves_yielded_controls_and_adds_baseline(monkeypatch):
    from powernap.capabilities import Capabilities
    daemon = object.__new__(Daemon)
    daemon.cfg = Config(dry_run=True, manage_cpu=False, manage_nvidia=False, manage_amdgpu=False)
    daemon.repository = SimpleNamespace(record_capabilities=lambda value: True)
    daemon.collector = SimpleNamespace(capabilities=None)
    daemon.controller = SimpleNamespace(yielded_targets={"external"})
    daemon.transition = SimpleNamespace(current=Profile.BALANCED)
    daemon.baseline = {}
    daemon.degraded_reason = None
    daemon.needs_reconcile = False
    monkeypatch.setattr("powernap.daemon.discover", lambda: Capabilities())
    daemon._after_resume()
    assert daemon.controller.yielded_targets == {"external"}
    assert daemon.needs_reconcile is True
    assert daemon.collector.capabilities == Capabilities()


def test_resume_database_failure_enters_degraded_mode(monkeypatch):
    from powernap.capabilities import Capabilities
    daemon = object.__new__(Daemon)
    daemon.cfg = Config(dry_run=True, manage_cpu=False, manage_nvidia=False, manage_amdgpu=False)
    daemon.repository = SimpleNamespace(record_capabilities=lambda value: (_ for _ in ()).throw(RuntimeError("db")))
    daemon.collector = SimpleNamespace(capabilities=None)
    daemon.controller = SimpleNamespace(yielded_targets=set())
    daemon.transition = SimpleNamespace(current=Profile.BALANCED)
    daemon.baseline = {}
    daemon.degraded_reason = None
    daemon.needs_reconcile = False
    monkeypatch.setattr("powernap.daemon.discover", lambda: Capabilities())
    daemon._after_resume()
    assert daemon.degraded_reason == "database unavailable"
