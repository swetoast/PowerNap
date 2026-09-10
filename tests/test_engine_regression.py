from powernap.config import Config
from powernap.engine import DecisionEngine, TransitionManager
from powernap.model import CPUState, Decision, GPUState, PriceContext, Profile, SystemState, ThermalState


def state(temp=30, gpu=None):
    return SystemState("now", 1, CPUState(5, 10, 0, 0, 0, 0, temp, "cpu"), (gpu,) if gpu else (), PriceContext())


def decision(profile, thermal=ThermalState.NORMAL):
    return Decision(0, 0, profile, profile, Profile.MAXIMUM, profile, thermal, "test")


def test_hot_gpu_caps_profile_even_when_cpu_is_cool():
    gpu = GPUState("gpu", "nvidia", utilization=90, temperature_c=80)
    result = DecisionEngine(Config()).decide(state(30, gpu))
    assert result.thermal_state == ThermalState.HOT
    assert result.recommended <= Profile.BALANCED


def test_promote_samples_one_allows_first_candidate():
    manager = TransitionManager(Config(promote_samples=1), Profile.BALANCED, now=0)
    assert manager.evaluate(decision(Profile.RESPONSIVE), now=1).allowed


def test_downshift_requires_residence_and_stability():
    cfg = Config(minimum_residence_seconds=10, relax_seconds=5)
    manager = TransitionManager(cfg, Profile.RESPONSIVE, now=0)
    assert not manager.evaluate(decision(Profile.ECO), now=2).allowed
    assert not manager.evaluate(decision(Profile.ECO), now=6).allowed
    assert manager.evaluate(decision(Profile.ECO), now=10).allowed


def test_critical_override_is_immediate():
    manager = TransitionManager(Config(), Profile.MAXIMUM, now=0)
    result = manager.evaluate(decision(Profile.ECO, ThermalState.CRITICAL), now=1)
    assert result.allowed and result.profile == Profile.ECO
