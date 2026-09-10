# Changelog

## 0.9.2

- Added conservative Balanced thermal ceiling when temperature telemetry is unavailable.
- Added thermal recovery hysteresis to prevent profile chatter near thresholds.
- Fixed interrupted 0.9.0 database migrations so they resume safely and avoid duplicate imports.
- Fixed first-cycle reconciliation retry behavior after required control failures.
- Rejected unsupported price providers and non-finite electricity prices during validation.
- Fixed malformed CPU frequency readings causing control-plan failures.
- Normalized NVML byte-string identifiers and names.
- Added CLI, configuration, migration, pricing, thermal recovery, and packaging regression coverage.
- Added a GitHub Actions branch-coverage floor of 65 percent.
- Expanded the regression suite to 50 tests and measured 69.58 percent branch coverage.

## 0.9.1

- Fixed disabled CPU and GPU managers still producing control operations.
- Fixed NVIDIA temperature telemetry and power-limit baseline restoration.
- Fixed external-change ownership so PowerNap does not yield after its own verified writes.
- Fixed manage mode so externally changed settings are reapplied while a profile remains active.
- Fixed first-cycle reconciliation and single-sample promotion.
- Fixed GPU thermal states participating in the safety ceiling.
- Fixed required-operation rollback and CPU control ordering.
- Fixed database event loss from timestamp collisions and added migration from the 0.9.0 schema.
- Fixed missing configuration files being silently ignored.
- Fixed price deduplication, overlap rejection, flat-price ranking, fallback, stale cache, and midnight lookahead.
- Added AMDGPU telemetry and corrected sysfs unit conversion.
- Added interruptible daemon shutdown waits and portable optional systemd paths.
- Expanded the regression suite from 9 to 38 tests.

## 0.9.0

- Added the long-running daemon and systemd readiness and watchdog notifications.
- Added SQLite samples, decisions, control events, reports, and retention.
- Added capability discovery for CPUFreq, EPP, powercap, NVIDIA, and AMDGPU.
- Added CPU governor, frequency ceiling, EPP, NVIDIA power limit, and AMDGPU performance-level planning.
- Added read-back verification, required-operation rollback, shutdown restoration, and external-change yielding.
- Added protected process minimum profiles.
- Added status, report, prices, check, capabilities, once, and run commands.
- Added fixture-based capability tests and persistence tests.
- Kept dry-run enabled in the supplied configuration.
