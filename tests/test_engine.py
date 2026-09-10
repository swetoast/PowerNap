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
    accepted = manager.evaluate(d, now=105)
    assert accepted.allowed
    manager.commit(accepted.profile, now=105)
    assert manager.current == Profile.RESPONSIVE


def test_price_trend_avoids_escalating_before_sharply_higher_prices():
    cpu = CPUState(1, 2, 0, 0, 0, 0, 35, "cpu")
    price = PriceContext(rank=0.1, future_rank=0.8, trend=0.7, fresh=True, complete=True, quality="fresh")
    decision = DecisionEngine(Config()).decide(state(cpu, price))
    assert decision.efficiency_preference == Profile.BALANCED


def test_price_trend_avoids_deep_saving_before_sharply_lower_prices():
    cpu = CPUState(1, 2, 0, 0, 0, 0, 35, "cpu")
    price = PriceContext(rank=0.9, future_rank=0.2, trend=-0.7, fresh=True, complete=True, quality="fresh")
    decision = DecisionEngine(Config()).decide(state(cpu, price))
    assert decision.efficiency_preference == Profile.BALANCED
