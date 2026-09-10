from collections import namedtuple
from pathlib import Path

from powernap.capabilities import AmdGPU, Capabilities
from powernap.config import Config
from powernap.runtime import Collector

Temp = namedtuple("Temp", "label current high critical")


def test_cpu_temperature_prefers_package_over_hotter_nvme(monkeypatch):
    monkeypatch.setattr("powernap.runtime.psutil.sensors_temperatures", lambda fahrenheit=False: {
        "nvme": [Temp("Composite", 80, 90, 95)],
        "coretemp": [Temp("Core 0", 45, 80, 100), Temp("Package id 0", 50, 80, 100)],
    })
    assert Collector(Config())._temperature() == (50.0, "coretemp:Package id 0")


def test_amd_telemetry_uses_documented_sysfs_units(tmp_path: Path):
    device = tmp_path / "0000:01:00.0"
    hwmon = device / "hwmon" / "hwmon0"
    hwmon.mkdir(parents=True)
    (device / "gpu_busy_percent").write_text("25")
    (device / "mem_busy_percent").write_text("10")
    (hwmon / "temp1_input").write_text("55000")
    (hwmon / "power1_average").write_text("42000000")
    gpu = AmdGPU(str(device), "0000:01:00.0", "auto", (), 100000000, 50000000, 150000000)
    result = Collector(Config(), Capabilities(amd_gpus=(gpu,)))._amd()[0]
    assert result.temperature_c == 55
    assert result.power_w == 42
    assert result.power_limit_w == 100


def test_disabled_gpu_collection_returns_empty():
    collector = Collector(Config(manage_nvidia=False, manage_amdgpu=False))
    assert collector._nvidia() == []
    assert collector._amd() == []
