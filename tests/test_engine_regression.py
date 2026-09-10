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
    accepted = manager.evaluate(decision(Profile.ECO), now=10)
    assert accepted.allowed
    manager.commit(accepted.profile, now=10)
    assert manager.current == Profile.ECO


def test_critical_override_is_immediate():
    manager = TransitionManager(Config(), Profile.MAXIMUM, now=0)
    result = manager.evaluate(decision(Profile.ECO, ThermalState.CRITICAL), now=1)
    assert result.allowed and result.profile == Profile.ECO
    assert manager.current == Profile.MAXIMUM
    manager.commit(result.profile, now=1)
    assert manager.current == Profile.ECO


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


def test_transition_state_round_trip_preserves_candidate_and_ages():
    cfg = Config(promote_samples=3)
    manager = TransitionManager(cfg, Profile.BALANCED, now=100)
    manager.evaluate(decision(Profile.RESPONSIVE), now=110)
    saved = manager.export_state(now=120)
    restored = TransitionManager.from_state(cfg, saved, now=500)
    assert restored.current == Profile.BALANCED
    assert restored.candidate == Profile.RESPONSIVE
    assert restored.candidate_samples == 1
    assert restored.candidate_since == 490
    assert restored.last_change == 480


def test_transition_state_rejects_invalid_persisted_values():
    manager = TransitionManager.from_state(Config(), {"current": "invalid"}, now=10)
    assert manager.current == Profile.BALANCED
    assert manager.candidate == Profile.BALANCED


def test_transition_state_clamps_negative_ages_and_samples():
    state_data = {"current":"eco","candidate":"maximum","candidate_samples":-4,"candidate_age_seconds":-8,"residence_age_seconds":-2}
    manager = TransitionManager.from_state(Config(), state_data, now=20)
    assert manager.current == Profile.ECO
    assert manager.candidate == Profile.MAXIMUM
    assert manager.candidate_samples == 0
    assert manager.candidate_since == 20
    assert manager.last_change == 20


def test_accepted_transition_is_only_a_proposal_until_committed():
    manager = TransitionManager(Config(promote_samples=1), Profile.BALANCED, now=0)
    proposal = manager.evaluate(decision(Profile.RESPONSIVE), now=1)
    assert proposal.allowed
    assert manager.current == Profile.BALANCED
    manager.commit(proposal.profile, now=2)
    assert manager.current == Profile.RESPONSIVE


def test_cpu_and_gpu_thermal_states_and_ceilings_are_reported_separately():
    gpu = GPUState("gpu", "nvidia", utilization=10, temperature_c=80)
    result = DecisionEngine(Config()).decide(state(40, gpu))
    assert result.cpu_thermal_state == ThermalState.NORMAL
    assert result.gpu_thermal_state == ThermalState.HOT
    assert result.cpu_safety_ceiling == Profile.MAXIMUM
    assert result.gpu_safety_ceiling == Profile.BALANCED
    assert result.safety_ceiling == Profile.BALANCED


def test_missing_gpu_temperature_does_not_reduce_known_cpu_ceiling():
    gpu = GPUState("gpu", "nvidia", utilization=10, temperature_c=None)
    result = DecisionEngine(Config()).decide(state(40, gpu))
    assert result.cpu_thermal_state == ThermalState.NORMAL
    assert result.gpu_thermal_state == ThermalState.UNKNOWN
    assert result.gpu_safety_ceiling == Profile.MAXIMUM
    assert result.safety_ceiling == Profile.MAXIMUM
