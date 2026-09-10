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
