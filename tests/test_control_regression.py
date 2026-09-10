from pathlib import Path

from powernap.capabilities import Capabilities, CPUFreqPolicy, NvidiaGPU
from powernap.control import Controller
from powernap.model import ControlOperation, OperationResult, Profile, ResultState


def make_policy(path: Path, current_max="2500000"):
    path.mkdir(parents=True)
    (path / "scaling_governor").write_text("ondemand")
    (path / "scaling_max_freq").write_text(current_max)
    return CPUFreqPolicy(str(path), (0,), "acpi-cpufreq", ("powersave", "ondemand", "schedutil", "performance"), "ondemand", 800000, int(current_max), 800000, 2500000, (), None)


def test_disabled_managers_produce_no_operations(tmp_path):
    policy = make_policy(tmp_path / "policy0")
    caps = Capabilities(cpu_policies=(policy,), nvidia_gpus=(NvidiaGPU(0, "gpu", "GPU", 100, 200, 200, 150),))
    assert Controller(caps, True, False, False, False).plan(Profile.ECO) == []


def test_downscale_changes_governor_before_frequency(tmp_path):
    policy = make_policy(tmp_path / "policy0")
    plan = Controller(Capabilities(cpu_policies=(policy,)), True, True, False, False).plan(Profile.ECO)
    assert plan[0].target.endswith("scaling_governor")
    assert plan[-1].target.endswith("scaling_max_freq")


def test_upscale_changes_frequency_before_governor(tmp_path):
    policy = make_policy(tmp_path / "policy0", "1000000")
    plan = Controller(Capabilities(cpu_policies=(policy,)), True, True, False, False).plan(Profile.MAXIMUM)
    assert plan[0].target.endswith("scaling_max_freq")


def test_successful_apply_updates_expected_state(tmp_path):
    target = tmp_path / "value"
    target.write_text("old")
    ctl = Controller(Capabilities(), False, False, False, False)
    result = ctl.apply_transaction([ControlOperation("file", str(target), "old", "new")])
    assert result[0].state == ResultState.APPLIED
    assert ctl.external_changes({str(target): "new"}) == {}


def test_required_failure_rolls_back_prior_write(tmp_path):
    first = tmp_path / "first"
    first.write_text("old")
    missing = tmp_path / "missing" / "value"
    ctl = Controller(Capabilities(), False, False, False, False)
    results = ctl.apply_transaction([
        ControlOperation("file", str(first), "old", "new"),
        ControlOperation("file", str(missing), None, "x"),
    ])
    assert results[1].state == ResultState.FAILED
    assert first.read_text() == "old"


def test_dry_run_never_writes(tmp_path):
    target = tmp_path / "value"
    target.write_text("old")
    ctl = Controller(Capabilities(), True, False, False, False)
    result = ctl.apply_transaction([ControlOperation("file", str(target), "old", "new")])
    assert result[0].state == ResultState.SIMULATED
    assert target.read_text() == "old"


def test_malformed_current_frequency_does_not_crash_planning(tmp_path):
    policy = make_policy(tmp_path / "policy0")
    (tmp_path / "policy0" / "scaling_max_freq").write_text("invalid")
    plan = Controller(Capabilities(cpu_policies=(policy,)), True, True, False, False).plan(Profile.ECO)
    assert any(item.target.endswith("scaling_max_freq") for item in plan)


def test_yield_is_scoped_to_only_the_conflicting_control(tmp_path):
    policy = make_policy(tmp_path / "policy0")
    ctl = Controller(Capabilities(cpu_policies=(policy,)), True, True, False, False)
    governor = str(tmp_path / "policy0" / "scaling_governor")
    ctl.yield_targets((governor,))
    plan = ctl.plan(Profile.ECO)
    assert all(item.target != governor for item in plan)
    assert any(item.target.endswith("scaling_max_freq") for item in plan)


def test_restore_does_not_overwrite_yielded_external_control(tmp_path):
    target = tmp_path / "value"
    target.write_text("external")
    ctl = Controller(Capabilities(), False, False, False, False)
    ctl.yield_targets((str(target),))
    assert ctl.restore({str(target): "baseline"}) == []
    assert target.read_text() == "external"


def powercap_fixture(tmp_path, enabled=True, current=65000000, minimum=10000000, maximum=95000000):
    from powernap.capabilities import PowerCapConstraint, PowerCapZone
    target = tmp_path / "constraint_0_power_limit_uw"
    target.write_text(str(current))
    constraint = PowerCapConstraint(str(target), "long_term", current, minimum, maximum, 28000000)
    zone = PowerCapZone(str(tmp_path), "package-0", enabled, (constraint,))
    return target, Capabilities(powercap_zones=(zone,))


def test_powercap_profile_targets_are_bounded_and_monotonic(tmp_path):
    target, capabilities = powercap_fixture(tmp_path)
    controller = Controller(capabilities, True, False, False, False, True)
    values = [int(controller.plan(profile)[0].requested_value) for profile in Profile]
    assert 10000000 <= values[0] < values[1] < values[2] < values[3] <= 95000000
    assert values[-1] == 95000000
    assert target.read_text() == "65000000"


def test_powercap_disabled_zone_is_not_controlled(tmp_path):
    _, capabilities = powercap_fixture(tmp_path, enabled=False)
    assert Controller(capabilities, True, False, False, False, True).plan(Profile.ECO) == []


def test_powercap_without_bounds_is_read_only(tmp_path):
    _, capabilities = powercap_fixture(tmp_path, minimum=None, maximum=None)
    assert Controller(capabilities, True, False, False, False, True).plan(Profile.ECO) == []


def test_powercap_dry_run_does_not_write(tmp_path):
    target, capabilities = powercap_fixture(tmp_path)
    controller = Controller(capabilities, True, False, False, False, True)
    result = controller.apply_transaction(controller.plan(Profile.ECO))[0]
    assert result.state == ResultState.SIMULATED
    assert target.read_text() == "65000000"


def test_powercap_write_verifies_and_restores_baseline(tmp_path):
    target, capabilities = powercap_fixture(tmp_path)
    controller = Controller(capabilities, False, False, False, False, True)
    baseline = dict(controller.expected_state)
    result = controller.apply_transaction(controller.plan(Profile.ECO))[0]
    assert result.state == ResultState.APPLIED
    assert target.read_text() != "65000000"
    restored = controller.restore(baseline)
    assert restored[0].state == ResultState.APPLIED
    assert target.read_text() == "65000000"


def test_powercap_external_change_can_yield_only_that_constraint(tmp_path):
    target, capabilities = powercap_fixture(tmp_path)
    controller = Controller(capabilities, True, False, False, False, True)
    target.write_text("70000000")
    changes = controller.external_changes()
    assert str(target) in changes
    controller.yield_targets(changes)
    assert controller.plan(Profile.ECO) == []


def test_transaction_summary_reports_optional_failure_as_partial():
    required = ControlOperation("file", "/required", "a", "b", True)
    optional = ControlOperation("file", "/optional", "a", "b", False)
    results = [
        OperationResult(required, "b", ResultState.APPLIED),
        OperationResult(optional, "a", ResultState.FAILED, "denied"),
    ]
    summary = Controller.transaction_summary(results)
    assert summary["state"] == "partially_applied"
    assert summary["failed_required"] == 0
    assert summary["failed_optional"] == 1


def test_transaction_summary_reports_required_failure():
    operation = ControlOperation("file", "/required", "a", "b", True)
    summary = Controller.transaction_summary([
        OperationResult(operation, "a", ResultState.FAILED, "denied")
    ])
    assert summary["state"] == "failed"
    assert summary["failed_required"] == 1


def test_transaction_summary_reports_dry_run():
    operation = ControlOperation("file", "/value", "a", "b", True)
    summary = Controller.transaction_summary([
        OperationResult(operation, "a", ResultState.SIMULATED)
    ])
    assert summary["state"] == "simulated"


def test_global_cpu_upscale_orders_all_limits_before_policy_changes(tmp_path):
    first = make_policy(tmp_path / "policy0", "1000000")
    second = make_policy(tmp_path / "policy1", "1000000")
    plan = Controller(Capabilities(cpu_policies=(first, second)), True, True, False, False).plan(Profile.MAXIMUM)
    names = [Path(item.target).name for item in plan]
    last_limit = max(i for i, name in enumerate(names) if name == "scaling_max_freq")
    first_governor = min(i for i, name in enumerate(names) if name == "scaling_governor")
    assert last_limit < first_governor


def test_global_cpu_downscale_orders_policy_changes_before_all_limits(tmp_path):
    first = make_policy(tmp_path / "policy0")
    second = make_policy(tmp_path / "policy1")
    plan = Controller(Capabilities(cpu_policies=(first, second)), True, True, False, False).plan(Profile.ECO)
    names = [Path(item.target).name for item in plan]
    last_governor = max(i for i, name in enumerate(names) if name == "scaling_governor")
    first_limit = min(i for i, name in enumerate(names) if name == "scaling_max_freq")
    assert last_governor < first_limit


def test_powercap_only_selects_one_named_long_term_constraint(tmp_path):
    from powernap.capabilities import PowerCapConstraint, PowerCapZone
    short = tmp_path / "constraint_0_power_limit_uw"; short.write_text("50000000")
    long = tmp_path / "constraint_1_power_limit_uw"; long.write_text("60000000")
    zone = PowerCapZone(str(tmp_path), "package-0", True, (
        PowerCapConstraint(str(short), "short_term", 50000000, 10000000, 90000000, 1000000),
        PowerCapConstraint(str(long), "long_term", 60000000, 10000000, 90000000, 28000000),
    ))
    plan = Controller(Capabilities(powercap_zones=(zone,)), True, False, False, False, True).plan(Profile.ECO)
    assert [item.target for item in plan] == [str(long)]


def test_failed_target_enters_exponential_retry_backoff(tmp_path, monkeypatch):
    target = tmp_path / "missing" / "value"
    clock = iter((100.0, 100.0, 103.0, 106.0))
    monkeypatch.setattr("powernap.control.time.monotonic", lambda: next(clock))
    ctl = Controller(Capabilities(), False, False, False, False)
    operation = ControlOperation("file", str(target), None, "new")
    result = ctl.apply_transaction([operation])
    assert result[0].state == ResultState.FAILED
    assert ctl.retry_not_before[str(target)] == 105.0
    ctl.capabilities = Capabilities()
    assert ctl.retry_not_before[str(target)] > 103.0


def test_success_clears_target_retry_backoff(tmp_path):
    target = tmp_path / "value"
    target.write_text("old")
    ctl = Controller(Capabilities(), False, False, False, False)
    ctl.failure_counts[str(target)] = 2
    ctl.retry_not_before[str(target)] = 999999999.0
    result = ctl.apply_transaction([ControlOperation("file", str(target), "old", "new")])
    assert result[0].state == ResultState.APPLIED
    assert str(target) not in ctl.failure_counts
    assert str(target) not in ctl.retry_not_before


def test_amd_plan_includes_bounded_power_cap_and_profile_mode(tmp_path):
    from powernap.capabilities import AmdGPU
    device = tmp_path / "0000:01:00.0"
    hwmon = device / "hwmon" / "hwmon0"
    hwmon.mkdir(parents=True)
    (device / "power_dpm_force_performance_level").write_text("auto")
    (device / "pp_power_profile_mode").write_text("BOOTUP_DEFAULT")
    (hwmon / "power1_cap").write_text("50000000")
    gpu = AmdGPU(str(device), "0000:01:00.0", "auto", ("BOOTUP_DEFAULT", "COMPUTE"), 50000000, 30000000, 90000000)
    plan = Controller(Capabilities(amd_gpus=(gpu,)), True, False, False, True).plan(Profile.MAXIMUM)
    targets = {Path(item.target).name: item.requested_value for item in plan}
    assert targets["power_dpm_force_performance_level"] == "high"
    assert targets["pp_power_profile_mode"] == "1"
    assert targets["power1_cap"] == "90000000"


def test_separate_gpu_profile_does_not_reduce_cpu_target(tmp_path):
    policy = make_policy(tmp_path / "policy0", "1000000")
    from powernap.capabilities import AmdGPU
    device = tmp_path / "gpu"
    device.mkdir()
    (device / "power_dpm_force_performance_level").write_text("auto")
    gpu = AmdGPU(str(device), "gpu", "auto", (), None, None, None)
    plan = Controller(Capabilities(cpu_policies=(policy,), amd_gpus=(gpu,)), True, True, False, True).plan(Profile.MAXIMUM, Profile.ECO)
    requested = {Path(item.target).name: item.requested_value for item in plan}
    assert requested["scaling_max_freq"] == policy.hw_max_khz
    assert requested["power_dpm_force_performance_level"] == "low"


def test_frequency_target_snaps_to_nearest_advertised_step(tmp_path):
    path = tmp_path / "policy0"
    policy = make_policy(path, "2501000")
    policy = CPUFreqPolicy(
        policy.path,
        policy.affected_cpus,
        policy.driver,
        policy.governors,
        policy.governor,
        policy.min_khz,
        policy.max_khz,
        policy.hw_min_khz,
        policy.hw_max_khz,
        policy.epp_available,
        policy.epp,
        (800000, 1800000, 2100000, 2501000),
    )
    plan = Controller(Capabilities(cpu_policies=(policy,)), True, True, False, False).plan(Profile.BALANCED)
    frequency = next(item for item in plan if item.target.endswith("scaling_max_freq"))
    assert frequency.requested_value == 2100000


def test_frequency_target_remains_continuous_when_steps_are_not_exposed(tmp_path):
    policy = make_policy(tmp_path / "policy0", "2501000")
    plan = Controller(Capabilities(cpu_policies=(policy,)), True, True, False, False).plan(Profile.BALANCED)
    frequency = next(item for item in plan if item.target.endswith("scaling_max_freq"))
    assert frequency.requested_value == 2075000
