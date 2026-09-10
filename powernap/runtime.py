from __future__ import annotations

import os
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from statistics import mean

import psutil

from .capabilities import Capabilities, read_int, read_text
from .config import Config
from .model import CPUState, GPUState, PriceContext, SystemState


class Collector:
    def __init__(self, cfg: Config, capabilities: Capabilities | None = None):
        self.cfg = cfg
        self.capabilities = capabilities or Capabilities()
        self.history: deque[float] = deque(maxlen=cfg.history_samples)

    def _temperature(self) -> tuple[float | None, str | None]:
        try:
            temperatures = psutil.sensors_temperatures(fahrenheit=False)
        except Exception:
            return None, None
        for group in ("coretemp", "k10temp", "zenpower", "cpu_thermal"):
            entries = temperatures.get(group, [])
            labeled = [
                entry for entry in entries
                if entry.current is not None and any(label in (entry.label or "").lower() for label in ("package", "tctl", "tdie"))
            ]
            selected = labeled[0] if labeled else next((entry for entry in entries if entry.current is not None), None)
            if selected is not None:
                return float(selected.current), f"{group}:{selected.label or 'default'}"
        return None, None

    def _nvidia(self) -> list[GPUState]:
        if not self.cfg.manage_nvidia:
            return []
        try:
            import pynvml
            pynvml.nvmlInit()
        except Exception:
            return []
        result = []
        try:
            for index in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                try:
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                except Exception:
                    utilization = None
                try:
                    encoder = float(pynvml.nvmlDeviceGetEncoderUtilization(handle)[0])
                    decoder = float(pynvml.nvmlDeviceGetDecoderUtilization(handle)[0])
                except Exception:
                    encoder = decoder = None
                try:
                    temperature = float(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
                except Exception:
                    temperature = None
                def metric(call, scale=1.0):
                    try:
                        return float(call(handle)) / scale
                    except Exception:
                        return None
                result.append(GPUState(
                    identity=str(pynvml.nvmlDeviceGetUUID(handle)),
                    vendor="nvidia",
                    utilization=float(utilization.gpu) if utilization else None,
                    memory_utilization=float(utilization.memory) if utilization else None,
                    encoder_utilization=encoder,
                    decoder_utilization=decoder,
                    temperature_c=temperature,
                    power_w=metric(pynvml.nvmlDeviceGetPowerUsage, 1000.0),
                    power_limit_w=metric(pynvml.nvmlDeviceGetPowerManagementLimit, 1000.0),
                ))
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
        return result

    def _amd(self) -> list[GPUState]:
        if not self.cfg.manage_amdgpu:
            return []
        result = []
        for gpu in self.capabilities.amd_gpus:
            device = Path(gpu.path)
            hwmon = next(iter(sorted((device / "hwmon").glob("hwmon*"))), None)
            utilization = read_int(device / "gpu_busy_percent")
            memory = read_int(device / "mem_busy_percent")
            temp = read_int(hwmon / "temp1_input") / 1000 if hwmon and read_int(hwmon / "temp1_input") is not None else None
            power = read_int(hwmon / "power1_average") if hwmon else None
            result.append(GPUState(
                identity=gpu.pci_id,
                vendor="amd",
                utilization=float(utilization) if utilization is not None else None,
                memory_utilization=float(memory) if memory is not None else None,
                temperature_c=temp,
                power_w=power / 1_000_000 if power is not None else None,
                power_limit_w=gpu.power_cap_uw / 1_000_000 if gpu.power_cap_uw is not None else None,
            ))
        return result

    def collect(self, price: PriceContext | None = None) -> SystemState:
        per_cpu = [float(item) for item in psutil.cpu_percent(interval=1.0, percpu=True)]
        average = mean(per_cpu) if per_cpu else 0.0
        peak = max(per_cpu, default=0.0)
        busy_ratio = sum(item >= self.cfg.busy_core_threshold for item in per_cpu) / max(1, len(per_cpu))
        self.history.append(average)
        sustained = sum(item >= 65 for item in self.history) / len(self.history)
        cores = psutil.cpu_count(logical=True) or 1
        try:
            load = os.getloadavg()[0] / cores
        except OSError:
            load = 0.0
        try:
            iowait = float(getattr(psutil.cpu_times_percent(interval=None), "iowait", 0.0))
        except Exception:
            iowait = 0.0
        temperature, source = self._temperature()
        gpus = tuple(self._nvidia() + self._amd())
        return SystemState(
            datetime.now().astimezone().isoformat(timespec="seconds"),
            time.monotonic_ns(),
            CPUState(average, peak, busy_ratio, iowait, load, sustained, temperature, source),
            gpus,
            price or PriceContext(),
        )
