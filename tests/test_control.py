from powernap.capabilities import CPUFreqPolicy
from powernap.control import preferred_governor
from powernap.model import Profile


def policy(governors):
    return CPUFreqPolicy("/x", (0,), "acpi-cpufreq", tuple(governors), "ondemand", 800000, 2501000, 800000, 2501000, (), None)


def test_acpi_cpufreq_profile_mapping():
    p = policy(["conservative", "ondemand", "powersave", "performance", "schedutil"])
    assert preferred_governor(p, Profile.ECO) == "powersave"
    assert preferred_governor(p, Profile.BALANCED) == "schedutil"
    assert preferred_governor(p, Profile.MAXIMUM) == "performance"
