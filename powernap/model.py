from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum, IntEnum
from typing import Any


class Profile(IntEnum):
    ECO = 0
    BALANCED = 1
    RESPONSIVE = 2
    MAXIMUM = 3


class ThermalState(str, Enum):
    NORMAL = "normal"
    WARM = "warm"
    HOT = "hot"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class ResultState(str, Enum):
    APPLIED = "applied"
    PARTIAL = "partially_applied"
    HELD = "held"
    SIMULATED = "simulated"
    FAILED = "failed"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class CPUState:
    average: float
    peak: float
    busy_ratio: float
    iowait: float
    load_1m_ratio: float
    sustained_ratio: float
    temperature_c: float | None
    temperature_source: str | None


@dataclass(frozen=True)
class GPUState:
    identity: str
    vendor: str
    utilization: float | None = None
    memory_utilization: float | None = None
    encoder_utilization: float | None = None
    decoder_utilization: float | None = None
    temperature_c: float | None = None
    power_w: float | None = None
    power_limit_w: float | None = None


@dataclass(frozen=True)
class PriceContext:
    current_sek_kwh: float | None = None
    rank: float | None = None
    future_rank: float | None = None
    trend: float | None = None
    fresh: bool = False
    provider: str | None = None
    cache_age_seconds: float | None = None
    quality: str = "unavailable"
    complete: bool = False
    coverage_ratio: float = 0.0
    gap_count: int = 0


@dataclass(frozen=True)
class SystemState:
    timestamp: str
    monotonic_ns: int
    cpu: CPUState
    gpus: tuple[GPUState, ...] = ()
    price: PriceContext = field(default_factory=PriceContext)


@dataclass(frozen=True)
class Decision:
    demand_score: float
    gpu_score: float
    demand_floor: Profile
    efficiency_preference: Profile
    safety_ceiling: Profile
    recommended: Profile
    thermal_state: ThermalState
    reason: str
    cpu_thermal_state: ThermalState = ThermalState.UNKNOWN
    gpu_thermal_state: ThermalState = ThermalState.UNKNOWN
    cpu_safety_ceiling: Profile = Profile.BALANCED
    gpu_safety_ceiling: Profile = Profile.MAXIMUM

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("demand_floor", "efficiency_preference", "safety_ceiling", "recommended"):
            data[key] = Profile(data[key]).name.lower()
        data["thermal_state"] = self.thermal_state.value
        data["cpu_thermal_state"] = self.cpu_thermal_state.value
        data["gpu_thermal_state"] = self.gpu_thermal_state.value
        data["cpu_safety_ceiling"] = self.cpu_safety_ceiling.name.lower()
        data["gpu_safety_ceiling"] = self.gpu_safety_ceiling.name.lower()
        return data


@dataclass(frozen=True)
class ControlOperation:
    adapter: str
    target: str
    old_value: str | float | int | None
    requested_value: str | float | int
    required: bool = True


@dataclass(frozen=True)
class OperationResult:
    operation: ControlOperation
    verified_value: str | float | int | None
    state: ResultState
    error: str | None = None
