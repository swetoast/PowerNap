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


def test_thermal_recovery_hysteresis_prevents_chatter():
    engine = DecisionEngine(Config(thermal_recovery_c=5))
    assert engine.decide(state(91)).thermal_state == ThermalState.CRITICAL
    assert engine.decide(state(87)).thermal_state == ThermalState.CRITICAL
    assert engine.decide(state(84)).thermal_state == ThermalState.HOT
    assert engine.decide(state(72)).thermal_state == ThermalState.HOT
    assert engine.decide(state(69)).thermal_state == ThermalState.WARM
    assert engine.decide(state(57)).thermal_state == ThermalState.WARM
    assert engine.decide(state(54)).thermal_state == ThermalState.NORMAL


def test_missing_temperature_resets_hysteresis():
    engine = DecisionEngine(Config())
    assert engine.decide(state(91)).thermal_state == ThermalState.CRITICAL
    assert engine.decide(state(None)).thermal_state == ThermalState.UNKNOWN
    assert engine.decide(state(84)).thermal_state == ThermalState.HOT


def test_missing_temperature_uses_conservative_balanced_ceiling():
    result = DecisionEngine(Config()).decide(state(None))
    assert result.thermal_state == ThermalState.UNKNOWN
    assert result.safety_ceiling == Profile.BALANCED
    assert result.recommended <= Profile.BALANCED
