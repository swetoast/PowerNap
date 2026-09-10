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

## Not verified in this environment

- Physical CPUFreq, EPP, AMDGPU, NVIDIA, or RAPL writes
- systemd watchdog behavior under the NAS service manager
- Live electricity-provider responses
- Operation on every Linux kernel, driver, CPU, and GPU combination
- The design target of 85 percent branch coverage. The current measured result is 83.90 percent.

The supplied configuration remains in dry-run mode. Physical control must be validated on each target system before dry-run is disabled.
## Privilege boundary

The current deployment remains a hardened root service because the supported sysfs controls require elevated local write access and a separate privileged helper has not yet completed real-hardware validation. The service unit restricts privileges, writable paths, address families, kernel access, home access, temporary files, executable memory, and restart behavior. A helper will not be introduced until its typed operation protocol, canonical path validation, bounds enforcement, readback, rollback, and hardware tests are complete.

