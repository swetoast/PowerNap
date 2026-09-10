from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return None


def read_int(path: Path) -> int | None:
    value = read_text(path)
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


@dataclass(frozen=True)
class CPUFreqPolicy:
    path: str
    affected_cpus: tuple[int, ...]
    driver: str | None
    governors: tuple[str, ...]
    governor: str | None
    min_khz: int | None
    max_khz: int | None
    hw_min_khz: int | None
    hw_max_khz: int | None
    epp_available: tuple[str, ...]
    epp: str | None


@dataclass(frozen=True)
class PowerCapZone:
    path: str
    name: str
    constraints: tuple[str, ...]


@dataclass(frozen=True)
class NvidiaGPU:
    index: int
    uuid: str
    name: str
    min_power_w: float | None
    max_power_w: float | None
    default_power_w: float | None
    current_power_w: float | None


@dataclass(frozen=True)
class AmdGPU:
    path: str
    pci_id: str
    performance_level: str | None
    profile_modes: tuple[str, ...]
    power_cap_uw: int | None
    power_min_uw: int | None
    power_max_uw: int | None


@dataclass(frozen=True)
class Capabilities:
    cpu_policies: tuple[CPUFreqPolicy, ...] = ()
    powercap_zones: tuple[PowerCapZone, ...] = ()
    nvidia_gpus: tuple[NvidiaGPU, ...] = ()
    amd_gpus: tuple[AmdGPU, ...] = ()
    kernel_cmdline: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _cpu_list(value: str | None) -> tuple[int, ...]:
    if not value:
        return ()
    cpus: list[int] = []
    for token in value.replace(",", " ").split():
        if "-" in token:
            try:
                start, end = map(int, token.split("-", 1))
                cpus.extend(range(start, end + 1))
            except ValueError:
                continue
        else:
            try:
                cpus.append(int(token))
            except ValueError:
                continue
    return tuple(dict.fromkeys(cpus))


def discover_cpu(root: Path = Path("/sys")) -> tuple[CPUFreqPolicy, ...]:
    paths = list((root / "devices/system/cpu/cpufreq").glob("policy*"))
    paths.sort(key=lambda p: int(p.name[6:]) if p.name[6:].isdigit() else 10**9)
    result = []
    for path in paths:
        result.append(CPUFreqPolicy(
            path=str(path),
            affected_cpus=_cpu_list(read_text(path / "affected_cpus")),
            driver=read_text(path / "scaling_driver"),
            governors=tuple((read_text(path / "scaling_available_governors") or "").split()),
            governor=read_text(path / "scaling_governor"),
            min_khz=read_int(path / "scaling_min_freq"),
            max_khz=read_int(path / "scaling_max_freq"),
            hw_min_khz=read_int(path / "cpuinfo_min_freq"),
            hw_max_khz=read_int(path / "cpuinfo_max_freq"),
            epp_available=tuple((read_text(path / "energy_performance_available_preferences") or "").split()),
            epp=read_text(path / "energy_performance_preference"),
        ))
    return tuple(result)


def discover_powercap(root: Path = Path("/sys")) -> tuple[PowerCapZone, ...]:
    result = []
    for path in sorted((root / "class/powercap").glob("*:*")):
        if path.is_dir():
            result.append(PowerCapZone(
                str(path),
                read_text(path / "name") or path.name,
                tuple(str(item) for item in sorted(path.glob("constraint_*_power_limit_uw"))),
            ))
    return tuple(result)


def _nvml_watts(call, handle) -> float | None:
    try:
        return float(call(handle)) / 1000.0
    except Exception:
        return None


def discover_nvidia() -> tuple[NvidiaGPU, ...]:
    try:
        import pynvml
        pynvml.nvmlInit()
    except Exception:
        return ()
    result = []
    try:
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            minimum = maximum = None
            try:
                minimum_mw, maximum_mw = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)
                minimum, maximum = minimum_mw / 1000.0, maximum_mw / 1000.0
            except Exception:
                pass
            result.append(NvidiaGPU(
                index=index,
                uuid=str(pynvml.nvmlDeviceGetUUID(handle)),
                name=str(pynvml.nvmlDeviceGetName(handle)),
                min_power_w=minimum,
                max_power_w=maximum,
                default_power_w=_nvml_watts(pynvml.nvmlDeviceGetPowerManagementDefaultLimit, handle),
                current_power_w=_nvml_watts(pynvml.nvmlDeviceGetPowerManagementLimit, handle),
            ))
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
    return tuple(result)


def discover_amd(root: Path = Path("/sys")) -> tuple[AmdGPU, ...]:
    result = []
    for card in sorted((root / "class/drm").glob("card[0-9]*")):
        device = card / "device"
        if read_text(device / "vendor") != "0x1002":
            continue
        hwmon = next(iter(sorted((device / "hwmon").glob("hwmon*"))), None)
        modes = []
        for line in (read_text(device / "pp_power_profile_mode") or "").splitlines():
            parts = line.replace("*", "").split()
            if len(parts) > 1 and parts[0].rstrip(":").isdigit():
                modes.append(parts[1])
        result.append(AmdGPU(
            path=str(device),
            pci_id=device.resolve().name,
            performance_level=read_text(device / "power_dpm_force_performance_level"),
            profile_modes=tuple(modes),
            power_cap_uw=read_int(hwmon / "power1_cap") if hwmon else None,
            power_min_uw=read_int(hwmon / "power1_cap_min") if hwmon else None,
            power_max_uw=read_int(hwmon / "power1_cap_max") if hwmon else None,
        ))
    return tuple(result)


def discover(root: Path = Path("/sys"), proc_root: Path = Path("/proc")) -> Capabilities:
    return Capabilities(
        cpu_policies=discover_cpu(root),
        powercap_zones=discover_powercap(root),
        nvidia_gpus=discover_nvidia(),
        amd_gpus=discover_amd(root),
        kernel_cmdline=read_text(proc_root / "cmdline") or "",
    )
