# Changelog

## 0.11.2

- Completed real-hardware CPUFreq validation across all twelve policies on the Ubuntu 22.04.5 target.
- Applied the advertised schedutil governor to every policy and verified every readback.
- Applied a bounded 2000000 kHz maximum-frequency ceiling to every policy and verified every readback.
- Restored each policy's captured original governor and maximum-frequency ceiling.
- Verified restoration independently on every policy.
- Kept boost, RAPL, and NVIDIA controls unchanged during the CPUFreq transaction.
- Kept dry-run enabled and power-cap control disabled by default.

## 0.11.1

- Recorded the first real-hardware CPUFreq validation on Ubuntu 22.04.5 with an Intel Core i5-12400 and the acpi-cpufreq driver.
- Verified a policy0 governor transition from ondemand to schedutil with successful readback.
- Verified automatic restoration from schedutil to the original ondemand governor with successful readback.
- Documented twelve independent CPUFreq policies, the 800000 to 2501000 kHz supported range, supported governors, and root write access.
- Documented that EPP is unavailable with the active acpi-cpufreq configuration.
- Documented package RAPL and NVIDIA capabilities discovered on the validation host without claiming those write paths as validated.
- Kept dry-run enabled and power-cap control disabled by default.

## 0.11.0

- Added non-blocking single-instance locking tied to the configured database path.
- Added PID recording in the active lock file and clean lock release during shutdown.
- Added a clear process exit code when another PowerNap daemon already owns the instance lock.
- Added deterministic cleanup for owned HTTP sessions while preserving injected test and caller sessions.
- Added HTTP-session cleanup to daemon and one-shot CLI lifecycles.
- Centralized the versioned status schema identifier.
- Added healthy-cycle watchdog, degraded-cycle watchdog, readiness, and stopping lifecycle tests.
- Added shutdown tests covering price, repository, and lock cleanup.
- Documented the hardened-root privilege boundary and the validation gate required before introducing a privileged helper.
- Expanded the regression suite to 119 tests.
- Raised daemon branch coverage to 88 percent.
- Raised measured branch coverage to 83.90 percent while retaining the 70 percent CI floor.

## 0.10.0

- Added labeled AMDGPU thermal selection with junction and hotspot priority.
- Added validation that rejects implausible GPU temperature values.
- Added bounded AMDGPU power-cap planning using discovered minimum and maximum limits.
- Added AMDGPU profile-mode planning with the real driver-reported numeric mode identifier.
- Added read-only Intel GPU discovery with duplicate DRM-device suppression.
- Added read-only Intel GPU utilization and temperature observation.
- Added independent CPU and GPU recommended profiles and controller targets.
- Prevented a low GPU target from reducing an independently required CPU target.
- Added regression fixtures for AMD labels, power caps, profile identifiers, Intel observation, and split control targets.
- Expanded the regression suite to 113 tests.
- Raised measured branch coverage to 82.91 percent while retaining the 70 percent CI floor.

## 0.9.9

- Added DST-aware complete-day validation using actual UTC duration between local midnights.
- Added explicit complete-day coverage, gap count, and expected-day duration fields to price context.
- Added verified 23-hour and 25-hour hourly DST fixtures.
- Added verified 92-interval and 100-interval quarter-hour DST fixtures.
- Added incomplete-day detection even when the immediate lookahead window is covered.
- Added negative, zero, unusually high finite, and out-of-order price regression tests.
- Fixed coverage calculations across DST boundaries by normalizing interval comparisons to UTC.
- Added systemd notification success, abstract-socket, missing-socket, and failure cleanup tests.
- Raised notification branch coverage to 100 percent.
- Expanded the regression suite to 107 tests.
- Raised measured branch coverage to 81.00 percent while retaining the 70 percent CI floor.

## 0.9.8

- Added first-class aggregate transaction state to status and report output.
- Added status schema version 1, applied profile, requested profile, and yielded-target reporting.
- Added full daemon-cycle tests proving verified success commits a physical transition and required failure does not.
- Added applied-profile reconstruction from the verified current control state.
- Added persistent external-change events for observe, yield, and manage policies.
- Added persistent yielded-target status.
- Added per-target exponential control retry backoff with reset after successful verification.
- Added database write retry backoff and automatic recovery-state clearing.
- Raised daemon branch coverage from 34 percent to 79 percent.
- Expanded the regression suite to 97 tests.
- Raised measured branch coverage to 79.80 percent while retaining the 70 percent CI floor.

## 0.9.7

- Added direction-aware global ordering across CPU policy operations so all frequency ceilings rise before policy changes and all policy changes precede frequency reductions.
- Limited RAPL control planning to one explicitly named long-term package constraint with valid reported bounds.
- Added AMDGPU discovery deduplication for multiple DRM cards pointing to the same PCI device.
- Added separate CPU and GPU thermal states, hysteresis tracking, and safety ceilings.
- Kept missing GPU temperature data from unnecessarily reducing a known-safe CPU ceiling.
- Added component thermal fields to structured decision output.
- Expanded the regression suite to 90 tests.
- Raised measured branch coverage to 75.09 percent while retaining the 70 percent CI floor.

## 0.9.6

- Added CPU sustained-load history warm-up so early samples cannot appear fully sustained.
- Added provider-isolated price contexts so current and future calculations never combine providers.
- Added interval coverage ratios, gap counts, completeness state, and incomplete quality reporting.
- Added duration-weighted future-price calculations for mixed hourly and quarter-hour intervals.
- Added fetching of every intermediate date covered by long lookahead windows.
- Added cross-midnight coverage evaluation using clipped interval durations.
- Added price-trend influence that avoids escalating immediately before sharply higher prices and avoids deep saving immediately before sharply lower prices.
- Expanded the regression suite to 84 tests.
- Raised measured branch coverage to 73.63 percent while retaining the 70 percent CI floor.

## 0.9.5

- Added restart-safe transition state for the current profile, candidate profile, candidate samples, stability age, and residence age.
- Added safe fallback when persisted transition state is missing, malformed, or invalid.
- Added explicit fresh, stale, and unavailable electricity-price quality states with cache age in seconds.
- Kept stale persisted prices visible while excluding them from price-weighted decisions.
- Added transaction summaries that distinguish applied, simulated, partially applied, and failed control cycles.
- Persisted the latest transaction summary for status and diagnostics.
- Recreated and hardened the systemd service unit while preserving required database and sysfs access.
- Recreated the repository .gitignore for Python, test, build, runtime, local configuration, and editor artifacts.
- Expanded the regression suite to 76 tests.
- Raised measured branch coverage to 71.84 percent while retaining the 70 percent CI floor.

## 0.9.4

- Added structured CPU inventory with vendor, model, logical CPU count, physical core count, and online CPU set.
- Added structured RAPL and power-cap constraints with current, minimum, maximum, time-window, zone-name, and enabled-state discovery.
- Added bounded profile-specific power-cap targets for Eco, Balanced, Responsive, and Maximum.
- Added power-cap dry-run isolation, read-back verification, rollback participation, baseline restoration, external-change detection, and per-control yielding.
- Added configuration checks that reject enabled power-cap control when safe bounded constraints are unavailable.
- Kept power-cap control disabled by default pending physical hardware validation.
- Expanded the regression suite to 68 tests.
- Raised measured branch coverage to 72.39 percent and the CI floor to 70 percent.

## 0.9.3

- Added persistent normalized electricity-price intervals in SQLite and restart-safe cache loading.
- Added capability snapshots with change fingerprints and retention.
- Added database health information to status and report output.
- Scoped yield ownership to only conflicting controls instead of stopping all management.
- Prevented shutdown restoration from overwriting externally yielded controls.
- Added long-gap and resume detection with capability rediscovery and transition reset.
- Preserved yielded controls and extended baseline coverage after capability refresh.
- Added a conservative Degraded status for required control failures and database failures.
- Added database recovery clearing after successful writes and guarded retention pruning.
- Expanded the regression suite to 57 tests.
- Measured 69.64 percent branch coverage.

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
