# Validation

## Automated checks completed

- Python bytecode compilation for every package module
- 38 regression tests covering configuration, capability fixtures, CPU and GPU control planning, control ordering, dry-run isolation, rollback, transitions, thermal ceilings, electricity-price intervals, provider fallback, stale cache behavior, SQLite migration, workload discovery, and telemetry mapping
- One-shot dry-run execution with JSON validation
- Capability command execution with JSON validation
- Configuration check execution with JSON validation
- Final ZIP integrity and content inspection

## Not verified in this environment

- Physical CPUFreq, EPP, AMDGPU, NVIDIA, or RAPL writes
- systemd watchdog behavior under the NAS service manager
- Live electricity-provider responses
- Operation on every Linux kernel, driver, CPU, and GPU combination
- The design target of 85 percent branch coverage because this execution environment does not include coverage.py or pytest-cov

The supplied configuration remains in dry-run mode. Physical control must be validated on each target system before dry-run is disabled.
