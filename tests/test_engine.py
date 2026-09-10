from powernap.config import Config
from powernap.engine import DecisionEngine, TransitionManager
from powernap.model import CPUState, Decision, PriceContext, Profile, SystemState, ThermalState


def state(cpu, price=PriceContext()):
    return SystemState("now", 1, cpu, (), price)


def test_single_busy_core_is_not_idle():
    cpu = CPUState(8, 95, 1/12, 0, .1, 0, 35, "coretemp:Package id 0")
    decision = DecisionEngine(Config()).decide(state(cpu))
    assert decision.demand_floor >= Profile.RESPONSIVE


def test_critical_temperature_forces_eco():
    cpu = CPUState(90, 100, 1, 0, 1, 1, 95, "cpu")
    decision = DecisionEngine(Config()).decide(state(cpu, PriceContext(rank=0, fresh=True)))
    assert decision.recommended == Profile.ECO
    assert decision.thermal_state == ThermalState.CRITICAL


def test_missing_price_defaults_balanced_but_demand_wins():
    cpu = CPUState(60, 90, .5, 0, .8, .5, 40, "cpu")
    decision = DecisionEngine(Config()).decide(state(cpu))
    assert decision.recommended >= Profile.RESPONSIVE


def test_transition_gradual_change_can_commit():
    cfg = Config(promote_samples=2)
    manager = TransitionManager(cfg, Profile.BALANCED)
    d = Decision(60, 0, Profile.RESPONSIVE, Profile.RESPONSIVE, Profile.MAXIMUM, Profile.RESPONSIVE, ThermalState.NORMAL, "test")
    assert not manager.evaluate(d, now=100).allowed
    assert manager.evaluate(d, now=105).allowed
