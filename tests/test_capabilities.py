from pathlib import Path
from powernap.capabilities import discover_cpu

def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(v)
def test_fixture_cpufreq(tmp_path:Path):
    p=tmp_path/"devices/system/cpu/cpufreq/policy0";write(p/"affected_cpus","0 1");write(p/"scaling_driver","acpi-cpufreq");write(p/"scaling_available_governors","powersave performance");write(p/"scaling_governor","powersave");write(p/"cpuinfo_min_freq","800000");write(p/"cpuinfo_max_freq","2500000")
    x=discover_cpu(tmp_path)[0];assert x.driver=="acpi-cpufreq" and x.affected_cpus==(0,1)


def test_powercap_discovery_reads_bounded_constraints(tmp_path):
    from powernap.capabilities import discover_powercap
    zone = tmp_path / "class" / "powercap" / "intel-rapl:0"
    zone.mkdir(parents=True)
    (zone / "name").write_text("package-0")
    (zone / "enabled").write_text("1")
    (zone / "constraint_0_name").write_text("long_term")
    (zone / "constraint_0_power_limit_uw").write_text("65000000")
    (zone / "constraint_0_min_power_uw").write_text("10000000")
    (zone / "constraint_0_max_power_uw").write_text("95000000")
    (zone / "constraint_0_time_window_us").write_text("28000000")
    zones = discover_powercap(tmp_path)
    assert len(zones) == 1
    assert zones[0].name == "package-0"
    assert zones[0].enabled is True
    constraint = zones[0].constraints[0]
    assert constraint.name == "long_term"
    assert constraint.power_limit_uw == 65000000
    assert constraint.min_power_uw == 10000000
    assert constraint.max_power_uw == 95000000
    assert constraint.time_window_us == 28000000


def test_powercap_discovery_tolerates_missing_optional_values(tmp_path):
    from powernap.capabilities import discover_powercap
    zone = tmp_path / "class" / "powercap" / "intel-rapl:0"
    zone.mkdir(parents=True)
    (zone / "constraint_0_power_limit_uw").write_text("invalid")
    result = discover_powercap(tmp_path)[0]
    assert result.enabled is None
    assert result.constraints[0].power_limit_uw is None


def test_cpu_inventory_discovers_vendor_model_cores_and_online_set(tmp_path):
    from powernap.capabilities import discover_cpu_info
    sysroot = tmp_path / "sys"
    procroot = tmp_path / "proc"
    write(sysroot / "devices/system/cpu/online", "0-2")
    write(procroot / "cpuinfo", """processor : 0
vendor_id : GenuineIntel
model name : Test CPU
physical id : 0
core id : 0

processor : 1
vendor_id : GenuineIntel
model name : Test CPU
physical id : 0
core id : 0

processor : 2
vendor_id : GenuineIntel
model name : Test CPU
physical id : 0
core id : 1
""")
    info = discover_cpu_info(sysroot, procroot)
    assert info.vendor == "GenuineIntel"
    assert info.model == "Test CPU"
    assert info.logical_cpus == 3
    assert info.physical_cores == 2
    assert info.online_cpus == (0, 1, 2)


def test_cpu_inventory_handles_missing_proc_data(tmp_path):
    from powernap.capabilities import discover_cpu_info
    info = discover_cpu_info(tmp_path / "sys", tmp_path / "proc")
    assert info.logical_cpus == 0
    assert info.physical_cores is None
    assert info.online_cpus == ()
