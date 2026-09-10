from pathlib import Path
from powernap.capabilities import discover_cpu

def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(v)
def test_fixture_cpufreq(tmp_path:Path):
    p=tmp_path/"devices/system/cpu/cpufreq/policy0";write(p/"affected_cpus","0 1");write(p/"scaling_driver","acpi-cpufreq");write(p/"scaling_available_governors","powersave performance");write(p/"scaling_governor","powersave");write(p/"scaling_available_frequencies","800000 1600000 2500000");write(p/"cpuinfo_min_freq","800000");write(p/"cpuinfo_max_freq","2500000")
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


def test_amdgpu_discovery_deduplicates_cards_for_same_device(tmp_path):
    from powernap.capabilities import discover_amd
    device = tmp_path / "devices" / "pci0000:00" / "0000:01:00.0"
    write(device / "vendor", "0x1002")
    drm = tmp_path / "class" / "drm"
    drm.mkdir(parents=True)
    for name in ("card0", "card1"):
        card = drm / name
        card.mkdir()
        (card / "device").symlink_to(device, target_is_directory=True)
    result = discover_amd(tmp_path)
    assert len(result) == 1
    assert result[0].pci_id == "0000:01:00.0"


def test_intel_gpu_discovery_is_read_only_and_deduplicated(tmp_path):
    from powernap.capabilities import discover_intel
    device = tmp_path / "devices" / "0000:00:02.0"
    write(device / "vendor", "0x8086")
    drm = tmp_path / "class" / "drm"
    drm.mkdir(parents=True)
    for name in ("card0", "card1"):
        card = drm / name
        card.mkdir()
        (card / "device").symlink_to(device, target_is_directory=True)
    result = discover_intel(tmp_path)
    assert len(result) == 1
    assert result[0].pci_id == "0000:00:02.0"


def test_amdgpu_discovery_preserves_profile_mode_ids(tmp_path):
    from powernap.capabilities import discover_amd
    device = tmp_path / "devices" / "0000:01:00.0"
    write(device / "vendor", "0x1002")
    write(device / "pp_power_profile_mode", "0 BOOTUP_DEFAULT\n5 COMPUTE *\n")
    card = tmp_path / "class" / "drm" / "card0"
    card.mkdir(parents=True)
    (card / "device").symlink_to(device, target_is_directory=True)
    gpu = discover_amd(tmp_path)[0]
    assert gpu.profile_modes == ("BOOTUP_DEFAULT", "COMPUTE")
    assert gpu.profile_mode_ids == (("BOOTUP_DEFAULT", "0"), ("COMPUTE", "5"))


def _write_cpu_topology(root, cpu, core, package=0):
    write(root / "devices/system/cpu" / f"cpu{cpu}" / "topology/core_id", str(core))
    write(root / "devices/system/cpu" / f"cpu{cpu}" / "topology/physical_package_id", str(package))


def test_cpu_inventory_prefers_arm_sysfs_topology_over_proc_count(tmp_path):
    from powernap.capabilities import discover_cpu_info
    sys_root = tmp_path / "sys"
    proc_root = tmp_path / "proc"
    write(sys_root / "devices/system/cpu/possible", "0-3")
    write(sys_root / "devices/system/cpu/present", "0-3")
    write(sys_root / "devices/system/cpu/online", "0-3")
    for cpu in range(4):
        _write_cpu_topology(sys_root, cpu, cpu)
    records = [f"processor : {cpu}\nCPU implementer : 0x41" for cpu in range(5)]
    write(proc_root / "cpuinfo", "\n\n".join(records))
    info = discover_cpu_info(sys_root, proc_root)
    assert info.logical_cpus == 4
    assert info.physical_cores == 4
    assert info.online_cpus == (0, 1, 2, 3)


def test_cpu_inventory_counts_unique_package_and_noncontiguous_core_ids(tmp_path):
    from powernap.capabilities import discover_cpu_info
    sys_root = tmp_path / "sys"
    proc_root = tmp_path / "proc"
    write(sys_root / "devices/system/cpu/present", "0-7")
    write(sys_root / "devices/system/cpu/online", "0-7")
    for cpu, core in enumerate((0, 0, 4, 4, 12, 13, 14, 15)):
        _write_cpu_topology(sys_root, cpu, core)
    write(proc_root / "cpuinfo", "vendor_id : GenuineIntel\nmodel name : test")
    info = discover_cpu_info(sys_root, proc_root)
    assert info.logical_cpus == 8
    assert info.physical_cores == 6


def test_cpu_inventory_counts_same_core_id_on_different_packages(tmp_path):
    from powernap.capabilities import discover_cpu_info
    sys_root = tmp_path / "sys"
    proc_root = tmp_path / "proc"
    write(sys_root / "devices/system/cpu/present", "0-1")
    write(sys_root / "devices/system/cpu/online", "0-1")
    _write_cpu_topology(sys_root, 0, 0, 0)
    _write_cpu_topology(sys_root, 1, 0, 1)
    write(proc_root / "cpuinfo", "processor : 0\n\nprocessor : 1")
    assert discover_cpu_info(sys_root, proc_root).physical_cores == 2


def test_cpu_inventory_keeps_present_and_online_counts_separate(tmp_path):
    from powernap.capabilities import discover_cpu_info
    sys_root = tmp_path / "sys"
    proc_root = tmp_path / "proc"
    write(sys_root / "devices/system/cpu/present", "0-3")
    write(sys_root / "devices/system/cpu/online", "0-2")
    for cpu in range(4):
        _write_cpu_topology(sys_root, cpu, cpu)
    write(proc_root / "cpuinfo", "processor : 0")
    info = discover_cpu_info(sys_root, proc_root)
    assert info.logical_cpus == 4
    assert info.physical_cores == 4
    assert info.online_cpus == (0, 1, 2)


def test_cpu_inventory_does_not_invent_physical_count_when_sysfs_topology_is_incomplete(tmp_path):
    from powernap.capabilities import discover_cpu_info
    sys_root = tmp_path / "sys"
    proc_root = tmp_path / "proc"
    write(sys_root / "devices/system/cpu/present", "0-1")
    write(sys_root / "devices/system/cpu/online", "0-1")
    _write_cpu_topology(sys_root, 0, 0)
    write(proc_root / "cpuinfo", "processor : 0\n\nprocessor : 1")
    info = discover_cpu_info(sys_root, proc_root)
    assert info.logical_cpus == 2
    assert info.physical_cores is None


def test_gpu_discovery_ignores_drm_connectors_and_returns_empty_without_supported_hardware(tmp_path):
    from powernap.capabilities import discover_amd, discover_intel
    drm = tmp_path / "class/drm"
    write(drm / "card1-HDMI-A-1" / "status", "connected")
    write(drm / "card1-DP-1" / "status", "disconnected")
    assert discover_amd(tmp_path) == ()
    assert discover_intel(tmp_path) == ()


def test_gpu_configuration_does_not_create_undiscovered_hardware(tmp_path):
    from powernap.capabilities import Capabilities
    from powernap.control import Controller
    from powernap.model import Profile
    capabilities = Capabilities()
    controller = Controller(capabilities, True, False, True, True, False)
    assert controller.plan(Profile.MAXIMUM) == []


def test_cpu_list_parser_handles_ranges_duplicates_and_invalid_tokens():
    from powernap.capabilities import _cpu_list
    assert _cpu_list("0-3,2,8-9 invalid 12") == (0, 1, 2, 3, 8, 9, 12)
    assert _cpu_list("broken-range") == ()
    assert _cpu_list(None) == ()


def test_cpu_inventory_uses_possible_when_present_is_unavailable(tmp_path):
    from powernap.capabilities import discover_cpu_info
    sys_root = tmp_path / "sys"
    proc_root = tmp_path / "proc"
    write(sys_root / "devices/system/cpu/possible", "0-1")
    for cpu in range(2):
        _write_cpu_topology(sys_root, cpu, cpu)
    write(proc_root / "cpuinfo", "processor : 0\n\nprocessor : 1\n\nprocessor : 2")
    info = discover_cpu_info(sys_root, proc_root)
    assert info.logical_cpus == 2
    assert info.physical_cores == 2
    assert info.online_cpus == (0, 1)


def test_cpu_inventory_falls_back_to_proc_topology_without_sysfs_inventory(tmp_path):
    from powernap.capabilities import discover_cpu_info
    proc_root = tmp_path / "proc"
    cpuinfo = "\n\n".join((
        "processor : 0\nphysical id : 0\ncore id : 0",
        "processor : 1\nphysical id : 0\ncore id : 0",
        "processor : 2\nphysical id : 0\ncore id : 1",
    ))
    write(proc_root / "cpuinfo", cpuinfo)
    info = discover_cpu_info(tmp_path / "sys", proc_root)
    assert info.logical_cpus == 3
    assert info.physical_cores == 2
    assert info.online_cpus == (0, 1, 2)


def test_nvml_helpers_handle_bytes_and_unavailable_power_values():
    from powernap.capabilities import _nvml_text, _nvml_watts
    assert _nvml_text(b"GPU-test") == "GPU-test"
    assert _nvml_text("GPU-test") == "GPU-test"
    assert _nvml_watts(lambda handle: 150000, object()) == 150.0
    assert _nvml_watts(lambda handle: (_ for _ in ()).throw(RuntimeError("unavailable")), object()) is None
