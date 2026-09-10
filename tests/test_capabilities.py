from pathlib import Path
from powernap.capabilities import discover_cpu

def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(v)
def test_fixture_cpufreq(tmp_path:Path):
    p=tmp_path/"devices/system/cpu/cpufreq/policy0";write(p/"affected_cpus","0 1");write(p/"scaling_driver","acpi-cpufreq");write(p/"scaling_available_governors","powersave performance");write(p/"scaling_governor","powersave");write(p/"cpuinfo_min_freq","800000");write(p/"cpuinfo_max_freq","2500000")
    x=discover_cpu(tmp_path)[0];assert x.driver=="acpi-cpufreq" and x.affected_cpus==(0,1)
