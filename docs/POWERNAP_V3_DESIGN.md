# PowerNap v3 Design Document

**Status:** Proposed design  
**Document type:** Architecture and behavior specification  
**Date:** 2026-09-10  
**Target platform:** Linux systems managed locally through a CLI and systemd service

## 1. Purpose

PowerNap v3 will evolve from a CPU governor switcher into a local, capability-aware Linux power-management daemon. It will observe workload demand, Swedish electricity prices, temperatures, hardware power telemetry, and administrator policy, then select and safely apply a coordinated system power profile.

The daemon must improve energy efficiency without making interactive tasks unreliable, interrupting sustained work, damaging hardware, or pretending unsupported controls are available. CPU governor selection remains supported, but it becomes only one control among CPU frequency limits, CPU energy preferences, package power caps, GPU profiles and power caps, and carefully selected device-runtime controls.

Linux exposes CPU performance scaling through CPUFreq policy objects and driver-specific interfaces such as `intel_pstate` and `amd_pstate`. The kernel also exposes a generic power-capping framework through sysfs, including Intel RAPL zones and constraints. These should be discovered at runtime rather than assumed from CPU vendor names.

## 2. Design goals

1. **Protect performance when demand is real.** Electricity price may influence spare performance and background work, but must not starve active, latency-sensitive, or explicitly protected workloads.
2. **Control actual power where supported.** Prefer measurable power caps, energy-performance preferences, and frequency ceilings over repeatedly switching governors.
3. **Treat hardware capabilities as authoritative.** Detect available drivers, policies, limits, sensors, and write permissions before enabling a control.
4. **Fail safely.** A failed read, stale price feed, missing sensor, or unsupported adapter must degrade to a conservative local policy, not an unsafe setting.
5. **Make every decision explainable.** Store observation, recommendation, gating result, attempted controls, verified result, and a concise reason.
6. **Avoid oscillation.** Use hysteresis, candidate duration, separate escalation and relaxation timing, and minimum profile residency.
7. **Keep the daemon local and small.** No web server, cloud command channel, remote-control API, or mandatory external service beyond the configured price provider.
8. **Remain useful offline.** Workload and thermal management continue without price data.
9. **Do not overreach.** Experimental or risky controls remain disabled unless explicitly enabled and validated.
10. **Support dry-run honestly.** Dry-run may simulate profile state, but can never record a physical change as applied.

## 3. Non-goals

PowerNap v3 will not:

- Replace the kernel scheduler, thermal framework, firmware protection, or hardware fan controller.
- Overclock, undervolt, edit firmware tables, or write undocumented registers.
- Force PCIe ASPM on unsupported hardware.
- Perform process termination, CPU affinity reassignment, or workload suspension in the initial design.
- Change filesystem writeback, swap, zram, or memory-reclaim parameters as part of power profiles.
- Automatically tune network interfaces, disks, or USB devices in the first implementation.
- Guarantee a specific watt saving, because available telemetry and controls vary by platform.
- Treat electricity spot price as the total household electricity cost.

PCIe power saving can trade latency for lower link power, and forcibly enabling ASPM on unsupported hardware can make a system unresponsive. PowerNap therefore may observe PCI runtime state, but global ASPM mutation is out of scope for the initial release.

## 4. Core operating principle

The processing pipeline is:

```text
Discover capabilities
        |
        v
Collect observations
        |
        v
Validate and normalize data
        |
        v
Build immutable system state
        |
        v
Calculate demand, price, and thermal constraints
        |
        v
Select an abstract power profile
        |
        v
Stabilize the candidate transition
        |
        v
Build a capability-specific control plan
        |
        v
Apply controls transactionally where possible
        |
        v
Read back and verify every changed control
        |
        v
Record recommendation, attempt, and verified result
```

Policy selects an abstract profile. Hardware adapters translate that profile into supported controls. The decision engine never writes sysfs, executes vendor tools, performs network requests, or queries SQLite.

## 5. Abstract power profiles

### 5.1 Eco

Purpose: minimize avoidable power use when demand is low or energy is expensive.

Expected behavior:

- Dynamic CPU scaling remains enabled.
- CPU energy preference favors efficiency.
- CPU maximum performance or frequency is reduced within configured safe bounds.
- CPU package power cap is lowered when supported.
- GPU remains in automatic or low-power mode.
- Discrete GPU power cap may be lowered, but never below the hardware-reported minimum.
- No latency-sensitive workload may be forced into Eco solely because of price.

### 5.2 Balanced

Purpose: normal unattended operation with reasonable responsiveness.

Expected behavior:

- Dynamic CPU scaling.
- Balanced energy-performance preference.
- Moderate CPU and GPU power limits.
- No unnecessary fixed maximum clocks.
- Default profile when price data is unavailable and no exceptional workload or thermal condition exists.

### 5.3 Responsive

Purpose: protect interactive latency, short bursts, media serving, and moderate compute work.

Expected behavior:

- Fast CPU response and a higher frequency ceiling.
- CPU package headroom increased within configured bounds.
- Active GPU receives adequate headroom.
- Price has reduced influence while the workload remains active.

### 5.4 Maximum

Purpose: provide full configured hardware performance for sustained, explicitly protected demand.

Expected behavior:

- Highest administrator-approved CPU and GPU limits.
- This does not mean bypassing manufacturer limits or thermal protection.
- Entry requires strong sustained demand, an explicit override, or a protected workload policy.
- Exit is delayed enough to avoid interrupting ongoing work.

### 5.5 Thermal Protect

Purpose: reduce heat immediately and independently of price.

Expected behavior:

- Overrides all normal profiles.
- Applies configured CPU and GPU caps in a deterministic order.
- Bypasses ordinary transition holds.
- Uses separate recovery hysteresis and a sustained cool period before leaving.
- Never disables firmware or kernel thermal mechanisms.

### 5.6 Degraded

Purpose: remain predictable when critical observations or controls are unreliable.

Examples:

- Inconsistent CPU policy state
- Repeated control verification failure
- Unusable thermal sensors when thermal enforcement is required
- Database unavailable

Degraded mode uses a configured safe profile, avoids controls whose state cannot be verified, emits a clear status, and periodically attempts recovery.

## 6. Capability discovery

Discovery runs at startup and can be refreshed after resume, driver reload, or hardware change.

### 6.1 CPU discovery

PowerNap records:

- CPU vendor, model, logical CPUs, physical cores, and online CPUs
- CPUFreq policy directories rather than duplicate per-CPU symlinks
- CPUs covered by each policy
- Active scaling driver
- Available and current governors per policy
- Minimum, maximum, current, and hardware-supported frequencies
- Boost support and current boost state, read-only initially
- Available energy-performance preference values
- Power-cap zones and constraint limits
- Read and write permissions

Modern CPU behavior is driver-specific. `intel_pstate`, `amd_pstate`, and generic CPUFreq governors must not be treated as equivalent, and the same governor name can have different practical behavior depending on the active driver.

### 6.2 GPU discovery

Each GPU is identified by stable PCI address and, where available, a vendor UUID. PowerNap records vendor, driver, integrated/discrete status, telemetry, supported controls, valid ranges, and whether a change survives reboot or driver reset.

#### AMD GPU adapter

Potential observations and controls:

- GPU busy percentage and memory busy percentage
- Temperature labels and limits
- Average or instantaneous power
- Reported power-cap minimum, maximum, and current setting
- DPM performance level
- Power profile mode
- Supported clocks as diagnostics

The AMDGPU driver documents hwmon temperature, power, fan, and clock telemetry, as well as sysfs power-state interfaces. On APUs, reported SoC power can include CPU consumption, so PowerNap must not add CPU and GPU readings together as if they were independent.

PowerNap will not control fan curves, overdrive voltage tables, or undocumented performance states.

#### NVIDIA GPU adapter

Preferred integration uses NVML bindings for stable programmatic access. `nvidia-smi` may be used for diagnostics or as a guarded fallback, but its text output is not a stable application interface. NVIDIA documents NVML and its Python bindings as backward-compatible choices for maintained tools.

Potential observations and controls:

- GPU and memory utilization
- Temperature
- Power draw
- Current, default, minimum, and maximum power limits
- Performance state
- Encoder and decoder activity when exposed
- Power-limit adjustment where supported and permitted

Unsupported values reported as `N/A` are capabilities not available, not zero measurements.

#### Intel integrated GPU adapter

Initial support is observation-first. PowerNap may consume standardized hwmon, DRM, or devfreq data when exposed. Writable controls are enabled only when a documented, discoverable interface and reliable read-back exist.

The kernel devfreq framework provides a CPUFreq-like mechanism for dynamic voltage and frequency scaling on non-CPU devices, but only devices whose drivers register a devfreq instance expose those controls.

### 6.3 Thermal sensor discovery

PowerNap builds a labeled sensor inventory and assigns sensors to components. It must not select the hottest sensor in the entire machine and call it CPU temperature.

CPU priority:

1. CPU package, Tctl, or equivalent package control temperature
2. CPU die or Tdie
3. Labeled physical core sensors
4. Explicitly configured hwmon channel
5. No CPU temperature, with a visible degraded thermal status

GPU temperatures come from the matching GPU adapter. NVMe, motherboard, and chipset sensors remain separate observations.

For each selected sensor, PowerNap stores:

- Stable sysfs path or adapter identity
- Human-readable label
- Component association
- Current reading
- Hardware critical or emergency threshold when exposed
- Last successful read time
- Selection reason

### 6.4 Device runtime power

The initial release observes PCI runtime state and may later support an explicit allowlist of runtime-PM devices. It will not apply a global `power/control=auto` sweep. Storage controllers, network interfaces, USB controllers, and GPUs can have wake, latency, or driver-specific constraints.

## 7. Electricity-price subsystem

### 7.1 Providers

The provider layer normalizes external responses into one internal price-point model. Initial providers:

1. `se.elpris.eu`, intended as the default compact Swedish provider after its request terms are reviewed and accepted for the deployment.
2. `elprisetjustnu.se`, supported as a direct provider and optional fallback.

`elpris.eu` describes its Swedish endpoint as a minimal JSON proxy for SE1 through SE4 aimed primarily at small devices. The direct Elpriset Just Nu API uses the static path form `YYYY/MM-DD_SEn.json`.

Provider selection is explicit. Automatic fallback is optional and must be logged, because silent provider switching complicates diagnostics and terms compliance.

### 7.2 Time resolution

The implementation must not assume 24 records per day. The source API states that Swedish data changed to quarter-hour prices from 1 October 2025, producing 96 intervals on ordinary days while older dates remain hourly.

PowerNap therefore uses interval timestamps from the response and supports:

- Historical hourly intervals
- Current quarter-hour intervals
- 23-hour and 25-hour DST days
- 92, 96, and 100 quarter-hour intervals on DST-affected days
- Missing, duplicated, overlapping, or out-of-order intervals

Day boundaries are calculated as local midnight to the next local midnight in the configured IANA timezone, never as `start + 86400`.

### 7.3 Normalized price point

Each normalized price interval contains:

- Provider
- Area, restricted to SE1 through SE4
- Interval start and end as timezone-aware values
- Start and end epoch timestamps
- Price in SEK/kWh
- Fetch timestamp
- Source date

Unknown fields are ignored at the provider boundary. Invalid intervals are rejected with a reason.

### 7.4 Price context

The price model produces continuous context instead of making `cheap`, `normal`, and `expensive` the primary decision values:

- Current price
- Daily minimum, maximum, median, and mean
- Percentile or robust rank within the local day
- Rolling average for configured future windows
- Relative change between now and future windows
- Availability of complete today and tomorrow data
- Age and validity of cached data
- Whether the current interval is among the day's cheapest or most expensive windows

Class labels may remain for reports, but profile selection consumes normalized continuous values.

### 7.5 Price influence

Price adjusts discretionary power headroom. It must not:

- Override Thermal Protect
- Reduce a protected workload below its minimum profile
- Cause a large profile change from one abnormal price interval
- Block a necessary performance escalation
- Treat unavailable data as expensive

When price data is missing or stale, its weight becomes zero and Balanced local policy continues.

## 8. Observation model

A cycle produces one immutable `SystemState` containing:

### CPU observations

- Average CPU utilization
- Maximum logical-core utilization
- Busy-core count and ratio
- CPU usage distribution percentiles
- One-, five-, and fifteen-minute load divided by online CPU capacity, without discarding overload above 1.0
- I/O wait
- Frequency and residency data when reliably exposed
- Package energy delta and derived power when available
- Selected CPU temperature and sensor identity

Median per-core utilization is retained only as one descriptive statistic. It cannot be the sole workload signal because a highly loaded single thread on a many-core machine can otherwise look idle.

### GPU observations

Per GPU:

- Utilization and memory utilization
- Video encode/decode activity where available
- Temperature
- Power draw and energy where available
- Current power cap/profile
- Active clients or workload presence where safely discoverable

### System observations

- Monotonic and wall-clock timestamps
- Online CPU count
- Suspend/resume discontinuity detection
- AC, battery, or UPS information when configured
- Current externally observed profile state
- Price context and quality
- Adapter health

## 9. Workload and demand model

The engine calculates independent scores from 0 to 100.

### 9.1 CPU demand

Signals include average utilization, peak-core utilization, busy-core ratio, normalized load queue, sustained history, and I/O wait. The model distinguishes:

- Idle
- Light background activity
- Single-thread burst
- Broad parallel burst
- Sustained compute
- I/O-bound activity
- Mixed workload

I/O wait is not treated as proof that more CPU power will help. It may justify responsiveness while preventing an unnecessary Maximum profile.

### 9.2 GPU demand

Signals include graphics/compute utilization, memory activity, video engines, recent active duration, and GPU power behavior. A transcoding workload can require GPU headroom even when CPU utilization is low.

### 9.3 Interactive demand

Optional signals may include recent local input activity or administrator-defined protected services. This feature is opt-in and local. PowerNap does not inspect document contents or network traffic.

### 9.4 Protected workloads

Administrators may define process, cgroup, systemd unit, or container rules that establish a minimum profile. Matching is explicit and auditable. No application names are hardcoded into PowerNap.

A protected workload rule can set:

- Minimum profile
- Required active duration before promotion
- Cooldown after the workload disappears
- CPU-only, GPU-only, or combined scope

## 10. Thermal model

Thermal decisions are component-specific.

States:

- Normal
- Warm
- Hot
- Critical
- Unknown

Thresholds may come from documented hardware limits or explicit configuration. User-configured thresholds must be validated in ascending order and kept below hardware emergency thresholds where known.

Behavior:

- Warm applies a soft ceiling or lower power target.
- Hot applies a stronger component cap and prevents profile escalation.
- Critical enters Thermal Protect immediately.
- Recovery requires both a lower exit threshold and a sustained cool interval.
- Unknown does not equal Normal. If a required sensor disappears, the adapter follows configured degraded behavior.

CPU and GPU thermal limits are applied independently. A hot GPU does not automatically force the CPU to minimum power unless shared package, APU telemetry, or system thermal policy requires it.

## 11. Profile selection

Profile selection uses ordered constraints instead of stacking repeated one-level adjustments.

1. Determine mandatory safety ceiling.
2. Determine minimum profile required by protected or measured demand.
3. Determine preferred efficiency profile from price and low-demand opportunity.
4. Clamp the preferred profile between the demand floor and safety ceiling.
5. Pass the candidate to the transition manager.

This prevents price from being counted in a baseline matrix, formula score, and a later explicit downgrade at the same time.

Conceptually:

```text
selected profile = clamp(efficiency preference, demand floor, safety ceiling)
```

The report includes all three inputs, so the operator can see whether demand, price, or temperature determined the outcome.

## 12. Transition control

The transition manager owns state across cycles.

### Escalation

- Critical thermal action is immediate.
- Strong new demand can promote after two consecutive valid samples.
- Moderate demand requires a longer candidate period.
- Price never delays an escalation required by protected demand.

### Relaxation

- Downshifts require a stable lower-profile candidate for a configurable duration.
- A minimum residence time prevents flapping.
- Recent burst activity extends the relaxation delay.
- Thermal recovery has separate hysteresis.

### Correct score comparison

Transition margins compare the current candidate with the state at the last accepted transition or with explicit enter/exit thresholds. They never compare only with the immediately previous loop score, which could block gradual legitimate changes indefinitely.

### External changes

Every cycle rereads managed settings at an appropriate interval. If another tool changes a managed control, PowerNap records an external change and follows configured ownership policy:

- `observe`: accept it and update state
- `manage`: reapply the selected PowerNap profile after a delay
- `yield`: stop managing that adapter until restart or operator action

Default ownership is `yield` for conflicting controls and `manage` only when explicitly configured.

## 13. Hardware control planning

Each profile expands into a `ControlPlan` containing only supported operations.

A CPU plan may include:

- Governor per CPUFreq policy
- Energy-performance preference
- Maximum performance percentage or frequency
- Minimum performance floor
- Package power-cap constraint

A GPU plan may include:

- Driver power profile
- Power cap
- Performance level, only when documented and safe

The plan stores old value, requested value, valid range, adapter, dependency order, rollback support, and verification method.

### Application order

For reducing power:

1. Reduce performance preference or dynamic target.
2. Reduce frequency/performance ceiling.
3. Reduce package or GPU power cap.
4. Apply governor only if still meaningful for the active driver.

For increasing performance:

1. Increase hard power cap.
2. Increase frequency/performance ceiling.
3. Increase energy-performance preference.
4. Apply governor if appropriate.

This avoids requesting higher clocks while the former low hard cap remains in place.

### Verification

Every write is followed by read-back. A profile is `applied` only when all required controls verify. Results can be:

- Applied
- Partially applied
- Held
- Simulated
- Failed
- Unsupported
- Externally changed

If an operation fails, PowerNap rereads all affected controls before deciding whether rollback is safe. It never assumes success from a zero exit status alone.

## 14. Dry-run semantics

Dry-run has separate values for:

- Observed physical profile
- Recommended profile
- Simulated profile
- Controls that would be attempted

Dry-run does not:

- Write sysfs
- Call a mutating vendor API
- Record a governor or power cap as physically applied
- Alter transition timestamps used for real operation

A dry-run history event is explicitly marked `simulated`.

## 15. Persistence

SQLite remains appropriate for local history. WAL mode may be used, but database errors are not converted into empty successful results.

Recommended logical tables:

### Price intervals

Normalized prices, provider, area, source interval, and fetch metadata.

### Observation samples

Periodic system-level summaries rather than one median row for every CPU. Per-core details are optional and sampled less frequently.

### Decisions

One row per evaluated decision with demand floor, efficiency preference, safety ceiling, candidate, gate result, and reason.

### Control events

One row per attempted control with old, requested, verified value, result, and error.

### Capability snapshots

Hardware and supported-control inventory recorded at startup and when it changes.

Retention is table-specific. Price intervals can be retained longer at low cost, while high-frequency samples are summarized before deletion.

## 16. Logging and status

Normal logs are concise and event-oriented. Continuous cycle details belong in structured debug logs or the database.

Required status information:

- Running mode
- Observed, recommended, candidate, and applied profiles
- Candidate age and hold reason
- CPU and GPU demand
- Price context and freshness
- Thermal state and selected sensors
- Active controls and verified values
- Adapter health
- Last successful price fetch
- Last control error

Reasons use direct language, for example:

```text
Responsive retained: GPU video workload is active; electricity price influence is suppressed until the workload ends.
```

## 17. CLI design

```text
powernap run
powernap once
powernap status
powernap check
powernap report
powernap prices
powernap capabilities
```

Important behavior:

- `once --dry-run` collects one complete state, prints the proposed plan and exits.
- `check` performs read-only validation by default.
- A separate explicit write test may apply and restore one supported control, with read-back at each step.
- `status --json` provides stable machine-readable output.
- `capabilities` explains why each adapter is supported, read-only, disabled, or unavailable.

## 18. Configuration principles

Configuration is validated before any control is applied.

Major sections:

```ini
[general]
[price]
[sampling]
[profiles]
[cpu]
[gpu]
[thermal]
[transitions]
[workloads]
[storage]
[logging]
```

Rules:

- Inline comments are parsed consistently for every type, including booleans.
- Invalid values name the exact section, key, supplied value, and expected range.
- Hardware limits are not guessed.
- Profile settings may be percentages where hardware absolute values vary.
- Absolute watt limits require validation against discovered supported ranges.
- Unsupported options produce warnings or errors according to strictness mode.
- Secrets are not required for the initial price providers.

## 19. Service lifecycle and security

PowerNap runs as a dedicated local systemd service. Because selected sysfs and vendor controls require privilege, the design should minimize its writable surface rather than granting unrestricted root behavior.

Preferred security model:

1. Discovery and decision logic run without broad privileges where practical.
2. A narrowly scoped privileged control boundary performs validated writes.
3. Allowed paths and value ranges come from discovered adapters, not arbitrary configuration paths.
4. No shell interpolation is used for control operations.
5. Vendor tools are invoked with fixed argument structures if a library interface is unavailable.
6. The service has no listening sockets.

The daemon supports systemd readiness and watchdog notifications. systemd expects watchdog keep-alives within the configured interval and recommends sending them at roughly half that interval.

Watchdog notification occurs only after a complete healthy cycle or a deliberate degraded cycle, not merely because the event loop is alive.

Shutdown behavior:

- Stop accepting new profile transitions.
- Finish or safely abort the active control plan.
- Flush pending history.
- Optionally restore configured baseline controls.
- Close databases and sessions.

## 20. Fault handling

### Price failure

Use valid cache within configured age. Otherwise set price weight to zero and continue local management.

### Sensor failure

Mark the sensor unknown, attempt rediscovery, and apply component-specific degraded policy.

### Control write failure

Read back actual state, record the failure, attempt safe rollback if supported, and prevent repeated rapid retries.

### Partial control application

Never label the full profile applied. Record the verified subset and move the adapter to partial or degraded state.

### Database failure

Continue safety-critical observation and control only if configured to do so. Log clearly that history is unavailable. Retry with backoff. Do not treat failed reads as empty valid datasets.

### Time jump or resume

Use monotonic time for durations. On suspend/resume or a large wall-clock jump, discard short-term sampling windows, refresh prices, rediscover controls where needed, and restart candidate timing.

## 21. Testing strategy

### Unit tests

- Configuration parsing and exact validation errors
- CPU and GPU demand scoring
- Price normalization and rank calculation
- Thermal state and recovery hysteresis
- Profile clamping by demand floor and safety ceiling
- Candidate timing and transition hysteresis
- Dry-run state separation

### Price tests

- Hourly historical day
- Ordinary 96-interval quarter-hour day
- 23-hour and 25-hour DST days
- 92 and 100 quarter-hour DST days
- Lookahead crossing midnight
- Missing tomorrow data
- Duplicate, overlapping, missing, and out-of-order intervals
- Negative, zero, and unusually high prices
- Provider fallback and stale cache

### CPU tests

- Single busy core on a many-core system
- Broad sustained workload
- Short burst
- I/O-bound load
- Load ratio above 1.0 preserved
- Multiple CPUFreq policies with different supported governors
- `intel_pstate`, `amd_pstate`, and generic CPUFreq fixtures
- External governor change
- Partial write failure

### GPU tests

- AMD dGPU telemetry and cap range
- AMD APU where SoC power includes CPU
- NVIDIA unsupported `N/A` fields
- NVIDIA power-cap range and read-back
- Multiple GPUs with stable identity
- Video engine active with low CPU usage
- Read-only Intel iGPU

### Thermal tests

- CPU sensor correctly selected over hotter NVMe
- Warm, hot, and critical entry
- Critical override bypasses holds
- Recovery requires hysteresis and duration
- Missing sensor enters configured degraded behavior
- GPU heat caps GPU independently where possible

### Integration tests

Use a fake sysfs tree and fake provider responses for deterministic tests. No automated test writes to real host power controls unless explicitly marked as hardware integration testing.

### Real-hardware validation

For every supported adapter:

1. Record baseline values.
2. Run capability discovery.
3. Perform read-only observation.
4. Run dry-run profile sequences.
5. Apply one bounded control.
6. Read back and verify.
7. Restore baseline.
8. Reboot and verify persistence assumptions.
9. Test suspend/resume where applicable.
10. Confirm thermal and workload behavior under controlled load.

No adapter is called supported until its final control path and restoration behavior have been tested on relevant hardware.

## 22. Implementation plan

### Phase 1: Correct baseline

- Repair escaped or malformed source text.
- Separate provider, collector, decision, transition, control, and persistence responsibilities.
- Correct dry-run behavior.
- Fix transition score gating.
- Implement timezone-safe interval handling.
- Introduce immutable normalized state models.
- Preserve the existing CPU-governor feature while adding tests.

### Phase 2: Capability-aware CPU management

- Discover CPUFreq policy objects and active driver.
- Add energy-performance preference and frequency-ceiling adapters.
- Add documented powercap/RAPL support where present.
- Implement abstract profiles and verified control plans.

### Phase 3: Price redesign

- Add `se.elpris.eu` and direct provider adapters.
- Support quarter-hour and DST-aware intervals.
- Implement price quality, cache age, lookahead, and continuous rank.
- Ensure offline operation removes price weight cleanly.

### Phase 4: GPU observation and control

- Add AMDGPU telemetry, profile, and bounded power-cap support.
- Add NVIDIA NVML telemetry and bounded power-cap support.
- Add Intel GPU observation where documented interfaces are available.
- Keep unsupported controls visibly read-only.

### Phase 5: Operational hardening

- Add status, capabilities, once, check, and JSON output.
- Add systemd readiness and watchdog support.
- Add privilege boundaries and service hardening.
- Add database migration, retention, and event auditing.

### Phase 6: Hardware validation

- Validate each adapter on real hardware.
- Document verified drivers, controls, and limitations.
- Tune defaults only from measured results.
- Do not enable experimental device controls by default.

## 23. Acceptance criteria

PowerNap v3 is ready for implementation completion only when:

- All profile decisions are deterministic from a supplied `SystemState`.
- Price handling supports hourly, quarter-hour, midnight crossover, and DST cases.
- A single busy core cannot be misclassified as idle solely because median utilization is low.
- CPU and GPU sensors are associated with the correct component.
- Dry-run performs no physical writes and records no fake applied event.
- Every applied control is read back and verified.
- Partial application is reported as partial, not successful.
- Unsupported hardware remains observable without unsafe guessed writes.
- Critical thermal transitions bypass normal holds.
- External control changes are detected.
- Price loss does not stop local management.
- Real-hardware tests restore original settings.
- Documentation identifies what was verified and what remains unverified.

## 24. Decisions fixed by this design

- PowerNap is profile-centric, not governor-centric.
- CPU, GPU, price, and thermal models remain separate until profile selection.
- Price influences discretionary headroom rather than overriding real demand.
- CPU governors are only one possible platform control.
- Actual watt caps are used only through documented, discoverable, range-checked interfaces.
- AMD APU SoC power is not double-counted as independent CPU plus GPU power.
- Global PCIe ASPM mutation is excluded from the initial release.
- Fan control, overclocking, undervolting, and undocumented registers are excluded.
- `se.elpris.eu` is the proposed default provider, subject to deployment acceptance of its terms; the direct API remains supported.
- The implementation must support Swedish quarter-hour prices and DST-variable interval counts.
- No code implementation should begin until this design, its validation rules, and test scope are accepted.

## 25. Research references

- Linux kernel power-management index and working-state CPU power management.
- Linux power-capping framework and sysfs topology.
- Linux AMDGPU power and thermal controls.
- NVIDIA management interfaces and NVML guidance.
- Linux devfreq framework.
- PCI power management and ASPM considerations.
- Swedish price providers and interval format.
- systemd watchdog behavior.
