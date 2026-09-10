from __future__ import annotations

import time
from dataclasses import dataclass

from .config import Config
from .model import Decision, Profile, SystemState, ThermalState


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


class DecisionEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def thermal_state(self, temperature: float | None) -> ThermalState:
        if temperature is None:
            return ThermalState.UNKNOWN
        if temperature >= self.cfg.critical_temp_c:
            return ThermalState.CRITICAL
        if temperature >= self.cfg.hot_temp_c:
            return ThermalState.HOT
        if temperature >= self.cfg.warm_temp_c:
            return ThermalState.WARM
        return ThermalState.NORMAL

    def decide(self, state: SystemState) -> Decision:
        cpu = state.cpu
        demand = clamp(
            cpu.average * 0.30
            + cpu.peak * 0.30
            + cpu.busy_ratio * 100 * 0.15
            + min(cpu.load_1m_ratio, 2.0) * 50 * 0.10
            + cpu.sustained_ratio * 100 * 0.15
        )
        gpu_score = max(
            (
                max(signal for signal in (gpu.utilization, gpu.memory_utilization, gpu.encoder_utilization, gpu.decoder_utilization) if signal is not None)
                for gpu in state.gpus
                if any(signal is not None for signal in (gpu.utilization, gpu.memory_utilization, gpu.encoder_utilization, gpu.decoder_utilization))
            ),
            default=0.0,
        )
        effective = max(demand, gpu_score)
        if effective >= 80 or (cpu.sustained_ratio >= 0.65 and effective >= 65):
            floor = Profile.MAXIMUM
        elif effective >= 45 or cpu.peak >= 80 or gpu_score >= 35:
            floor = Profile.RESPONSIVE
        elif effective >= 15:
            floor = Profile.BALANCED
        else:
            floor = Profile.ECO

        rank = state.price.rank if state.price.fresh else None
        if rank is None:
            preference = Profile.BALANCED
        elif rank >= 0.80:
            preference = Profile.ECO
        elif rank <= 0.25:
            preference = Profile.RESPONSIVE
        else:
            preference = Profile.BALANCED

        component_states = [self.thermal_state(cpu.temperature_c)]
        component_states.extend(self.thermal_state(gpu.temperature_c) for gpu in state.gpus)
        known = [item for item in component_states if item != ThermalState.UNKNOWN]
        thermal = max(known, key=lambda item: list(ThermalState).index(item)) if known else ThermalState.UNKNOWN
        ceiling = {
            ThermalState.CRITICAL: Profile.ECO,
            ThermalState.HOT: Profile.BALANCED,
            ThermalState.WARM: Profile.RESPONSIVE,
            ThermalState.NORMAL: Profile.MAXIMUM,
            ThermalState.UNKNOWN: Profile.MAXIMUM,
        }[thermal]
        recommended = Profile(min(max(int(preference), int(floor)), int(ceiling)))
        if thermal == ThermalState.CRITICAL:
            reason = "Thermal Protect requires Eco immediately."
        elif recommended == floor and floor > preference:
            reason = f"{floor.name.title()} required by measured demand; price cannot reduce it."
        elif recommended == ceiling and ceiling < preference:
            reason = f"{ceiling.name.title()} is the thermal safety ceiling."
        else:
            reason = f"{recommended.name.title()} selected from current demand and price opportunity."
        return Decision(round(demand, 1), round(gpu_score, 1), floor, preference, ceiling, recommended, thermal, reason)


@dataclass(frozen=True)
class TransitionResult:
    profile: Profile
    allowed: bool
    reason: str


class TransitionManager:
    def __init__(self, cfg: Config, initial: Profile = Profile.BALANCED, now: float | None = None):
        current_time = time.monotonic() if now is None else now
        self.cfg = cfg
        self.current = initial
        self.candidate = initial
        self.candidate_since = current_time
        self.candidate_samples = 0
        self.last_change = current_time

    def evaluate(self, decision: Decision, now: float | None = None) -> TransitionResult:
        current_time = time.monotonic() if now is None else now
        desired = decision.recommended
        if decision.thermal_state == ThermalState.CRITICAL:
            changed = desired != self.current
            self._commit(desired, current_time)
            return TransitionResult(desired, changed, "critical thermal override")
        if desired == self.current:
            self.candidate = desired
            self.candidate_samples = 0
            self.candidate_since = current_time
            return TransitionResult(self.current, False, "profile unchanged")
        if desired != self.candidate:
            self.candidate = desired
            self.candidate_samples = 1
            self.candidate_since = current_time
        else:
            self.candidate_samples += 1

        if desired > self.current:
            if self.candidate_samples < self.cfg.promote_samples:
                return TransitionResult(self.current, False, "promotion awaiting consecutive samples")
        else:
            if current_time - self.last_change < self.cfg.minimum_residence_seconds:
                return TransitionResult(self.current, False, "minimum profile residence active")
            if current_time - self.candidate_since < self.cfg.relax_seconds:
                return TransitionResult(self.current, False, "relaxation candidate not stable long enough")
        self._commit(desired, current_time)
        return TransitionResult(desired, True, "transition accepted")

    def _commit(self, profile: Profile, now: float) -> None:
        self.current = profile
        self.candidate = profile
        self.last_change = now
        self.candidate_since = now
        self.candidate_samples = 0
