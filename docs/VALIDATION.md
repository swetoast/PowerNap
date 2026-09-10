# Validation

## Automated checks completed

- Python bytecode compilation for every package module
- 119 regression tests covering configuration, capability fixtures, CLI behavior, CPU and GPU control planning, control ordering, dry-run isolation, rollback, transitions, thermal ceilings and recovery, electricity-price intervals, provider fallback, stale cache behavior, SQLite migration, workload discovery, packaging, and telemetry mapping
- Measured 83.90 percent branch coverage with a 70 percent CI minimum
- Clean virtual-environment installation and console entry-point smoke test
- One-shot installed-package dry-run with JSON validation
- Capability command execution with JSON validation
- Configuration check execution with JSON validation
- Final ZIP integrity and content inspection

## Real-hardware validation

Validation host:

- Ubuntu 22.04.5 LTS on bare metal
- Linux kernel 6.2.16-060216-generic
- Intel Core i5-12400 with 12 logical CPUs
- `acpi-cpufreq` driver with twelve independent policy directories
- Supported governors: conservative, ondemand, userspace, powersave, performance, and schedutil
- Supported frequency range: 800000 to 2501000 kHz
- CPU package temperature source: coretemp Package id 0
- NVIDIA GeForce GTX TITAN X with driver 580.178.04

Verified CPU governor transaction on policy0:

1. Captured the original governor `ondemand`.
2. Applied advertised governor `schedutil`.
3. Read back `schedutil` successfully.
4. Restored the original governor `ondemand`.
5. Read back `ondemand` successfully.

This verifies one bounded CPUFreq governor write, readback, and restoration path on this specific host. It does not by itself validate all twelve policies, frequency-ceiling writes, service-driven application, restart restoration, or other CPUFreq drivers.

Additional read-only hardware evidence:

- Root has write permission for CPUFreq governor, minimum frequency, maximum frequency, and boost controls.
- Boost is exposed at `/sys/devices/system/cpu/cpufreq/boost` and is enabled.
- EPP is not exposed with the active `acpi-cpufreq` configuration and is treated as unsupported.
- Package RAPL long-term constraint is present at 65000000 microwatts and root-writable, but no reliable minimum bound was exposed. RAPL control therefore remains disabled.
- NVIDIA NVML reports a 150 to 275 W supported power-limit range, with a current limit of 150 W and default limit of 250 W. NVIDIA physical writes remain unverified.
- No AMDGPU or active Intel GPU was present on the validation host.

## Not yet verified

- CPU frequency-ceiling write, readback, and restoration
- Governor application and restoration across all twelve CPUFreq policies
- EPP writes because the interface is unavailable on this host
- RAPL writes because no trustworthy minimum bound was discovered
- NVIDIA power-limit writes and restoration
- AMDGPU controls because no AMDGPU is installed
- Intel GPU observation on real hardware because no active Intel GPU is present
- Suspend and resume capability refresh on the target host
- Restore and reboot behavior
- systemd watchdog behavior under the target service manager
- Live electricity-provider responses
- Operation on every Linux kernel, driver, CPU, and GPU combination
- The design target of 85 percent branch coverage. The current measured result is 83.90 percent.

The supplied configuration remains in dry-run mode. Physical control must be validated on each target system before dry-run is disabled.

## Privilege boundary

The current deployment remains a hardened root service because the supported sysfs controls require elevated local write access and a separate privileged helper has not yet completed real-hardware validation. The service unit restricts privileges, writable paths, address families, kernel access, home access, temporary files, executable memory, and restart behavior. A helper will not be introduced until its typed operation protocol, canonical path validation, bounds enforcement, readback, rollback, and hardware tests are complete.
