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


def test_database_write_backoff_and_recovery(monkeypatch):
    daemon = object.__new__(Daemon)
    daemon.database_failure_count = 0
    daemon.database_retry_not_before = 0.0
    daemon.degraded_reason = None
    times = iter((100.0, 101.0, 106.0))
    monkeypatch.setattr("powernap.daemon.time.monotonic", lambda: next(times))
    failing = lambda: (_ for _ in ()).throw(RuntimeError("db"))
    assert daemon._repository_write(failing) is False
    assert daemon.degraded_reason == "database unavailable"
    assert daemon.database_retry_not_before == 105.0
    called = []
    assert daemon._repository_write(lambda: called.append(True)) is False
    assert daemon._repository_write(lambda: called.append(True)) is True
    assert called == [True]
    assert daemon.degraded_reason is None


def test_external_yield_event_is_persisted():
    daemon = bare_daemon("yield")
    events = []
    metadata = {}
    daemon.repository = SimpleNamespace(
        record_event=lambda kind, payload: events.append((kind, payload)),
        set_meta=lambda key, value: metadata.__setitem__(key, value),
    )
    daemon.database_failure_count = 0
    daemon.database_retry_not_before = 0.0
    daemon.degraded_reason = None
    assert daemon._resolve_external_changes() is False
    assert events[0][0] == "external_change"
    assert events[0][1]["policy"] == "yield"
    assert metadata["yielded_targets"] == ["x"]


def _cycle_daemon(tmp_path, result_state):
    from powernap.engine import DecisionEngine, TransitionManager
    from powernap.model import CPUState, ControlOperation, OperationResult, PriceContext, SystemState
    cfg = Config(
        dry_run=False,
        manage_cpu=False,
        manage_nvidia=False,
        manage_amdgpu=False,
        manage_powercap=False,
        promote_samples=1,
        restore_on_exit=False,
        sample_interval_sec=0.01,
        database_path=str(tmp_path / "cycle.db"),
    )
    daemon = object.__new__(Daemon)
    daemon.cfg = cfg
    daemon.stop_requested = False
    daemon.engine = DecisionEngine(cfg)
    daemon.transition = TransitionManager(cfg, Profile.BALANCED, now=0)
    daemon.simulated_transition = TransitionManager(cfg, Profile.BALANCED, now=0)
    daemon.price = SimpleNamespace(context=lambda lookahead_hours: PriceContext())
    state = SystemState("now", 1, CPUState(60, 90, 0.5, 0, 0.5, 0, 40, "cpu"), (), PriceContext())
    daemon.collector = SimpleNamespace(collect=lambda price: state)
    operation = ControlOperation("file", "/fake", "old", "new", True)
    result = OperationResult(operation, "new" if result_state.value == "applied" else "old", result_state, None if result_state.value == "applied" else "failed")
    daemon.controller = SimpleNamespace(
        yielded_targets=set(),
        plan=lambda profile: [operation],
        apply_transaction=lambda operations: [result],
        transaction_summary=lambda results: {
            "state": "applied" if result_state.value == "applied" else "failed",
            "operations": 1,
            "counts": {},
            "failed_required": 0 if result_state.value == "applied" else 1,
            "failed_optional": 0,
            "rollback_failures": 0,
        },
        external_changes=lambda current=None: {},
        snapshot=lambda: {"/fake": "new"},
        infer_applied_profile=lambda: Profile.RESPONSIVE if result_state.value == "applied" else Profile.BALANCED,
        restore=lambda baseline: [],
    )
    records = {"meta": {}, "cycles": []}
    daemon.repository = SimpleNamespace(
        record_cycle=lambda state, decision, results: records["cycles"].append((decision, results)),
        set_meta=lambda key, value: records["meta"].__setitem__(key, value),
        record_event=lambda kind, payload: None,
        prune=lambda samples, events: None,
        close=lambda: None,
    )
    daemon.baseline = {}
    daemon.needs_reconcile = True
    daemon.degraded_reason = None
    daemon.last_cycle_monotonic = None
    daemon.database_failure_count = 0
    daemon.database_retry_not_before = 0.0
    return daemon, records


def test_full_daemon_cycle_commits_only_after_verified_success(tmp_path, monkeypatch):
    from powernap.model import ResultState
    daemon, records = _cycle_daemon(tmp_path, ResultState.APPLIED)
    monkeypatch.setattr("powernap.daemon.signal.signal", lambda *args: None)
    monkeypatch.setattr("powernap.daemon.sd_notify", lambda message: None)
    daemon.run(max_cycles=1)
    assert daemon.transition.current == Profile.RESPONSIVE
    assert records["meta"]["last_transaction"]["state"] == "applied"
    assert records["meta"]["applied_profile"] == "responsive"


def test_full_daemon_cycle_required_failure_does_not_commit(tmp_path, monkeypatch):
    from powernap.model import ResultState
    daemon, records = _cycle_daemon(tmp_path, ResultState.FAILED)
    monkeypatch.setattr("powernap.daemon.signal.signal", lambda *args: None)
    monkeypatch.setattr("powernap.daemon.sd_notify", lambda message: None)
    daemon.run(max_cycles=1)
    assert daemon.transition.current == Profile.BALANCED
    assert daemon.needs_reconcile is True
    assert daemon.degraded_reason == "required control verification failed"
    assert records["meta"]["last_transaction"]["state"] == "failed"
