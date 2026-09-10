# PowerNap

PowerNap is a local, capability-aware Linux power-management daemon. It balances workload demand, electricity prices, and thermal conditions, then applies the safest power controls exposed by the running kernel and hardware drivers.

PowerNap is profile-centric rather than governor-centric. It selects an abstract operating profile, then maps that profile to the controls the current system actually supports. A machine may use CPUFreq governors and frequency ceilings, another may expose energy-performance preferences, and a GPU may provide a bounded power limit. Unsupported controls are reported and skipped instead of guessed.

> **Project status:** PowerNap 0.11.6 is a pre-release intended for dry-run validation and hardware testing. The supplied configuration has `dry_run = true`. CPU governor and frequency-ceiling writes across all twelve acpi-cpufreq policies, plus NVIDIA power-limit write, readback, and restoration through NVML, have been validated on the target Ubuntu host. Other physical control paths remain under validation.

## Table of contents

- [Why PowerNap](#why-powernap)
- [Features](#features)
- [Supported hardware and controls](#supported-hardware-and-controls)
- [How it works](#how-it-works)
- [Safety model](#safety-model)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [Command reference](#command-reference)
- [Example decisions](#example-decisions)
- [Running as a systemd service](#running-as-a-systemd-service)
- [Data and privacy](#data-and-privacy)
- [Testing](#testing)
- [Validation status](#validation-status)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Security](#security)
- [License](#license)

## Why PowerNap

Linux power-management tools often focus on one part of the system, such as CPU frequency, laptop battery life, thermal control, or a vendor-specific GPU interface. PowerNap brings those signals together in one local decision process.

PowerNap is designed to answer four questions continuously:

1. How much CPU and GPU performance does the current workload need?
2. Is electricity relatively cheap or expensive right now?
3. Is temperature limiting the safe performance ceiling?
4. Which documented controls are actually available on this machine?

Electricity price influences discretionary headroom. It does not override thermal protection or reduce a real workload below its measured or configured minimum profile.

## Features

- Abstract Eco, Balanced, Responsive, and Maximum profiles
- Immediate thermal protection for critical conditions
- CPU demand calculated from average use, peak-core use, busy-core ratio, normalized load, and warmed-up sustained activity
- GPU demand from utilization, memory activity, and video engines when available
- Swedish electricity prices through Elpris.eu with Elpriset Just Nu as an optional fallback
- Hourly and quarter-hour price intervals with provider isolation, DST-aware complete-day validation, gap detection, and duration-weighted lookahead
- Capability discovery before control planning
- Multiple CPUFreq policy support with direction-aware global operation ordering
- CPU governor, frequency ceiling, and energy-performance preference planning
- NVIDIA telemetry and hardware-bounded power limits through NVML
- AMDGPU discovery and documented performance-level control
- Powercap and RAPL discovery
- Separate CPU and GPU thermal states and safety ceilings
- Protected-process minimum profiles
- Candidate-based transitions with faster promotion, delayed relaxation, and restart-safe state
- External-setting conflict detection with persisted yield and policy events
- Read-back verification after changes
- Required-operation rollback after failure
- Baseline restoration on shutdown
- SQLite history for samples, decisions, control events, capability snapshots, price intervals, and transition state
- Versioned JSON status and report output with aggregate transaction state
- systemd readiness and watchdog notifications
- Dry-run mode enabled by default
- No web server or remote-control interface

## Supported hardware and controls

### CPU

PowerNap discovers generic Linux CPUFreq policy directories and records the active driver, available governors, policy CPU membership, current limits, hardware limits, and energy-performance preferences when exposed.

Expected discovery paths include:

- `acpi-cpufreq`
- `intel_pstate`
- `amd_pstate`
- Other drivers using the standard CPUFreq policy interface

Available control depends on the driver and kernel configuration:

- Scaling governor
- Maximum scaling frequency
- Energy-performance preference
- Powercap or RAPL discovery

PowerNap does not assume that every system exposes RAPL, EPP, or the same governor names.

### NVIDIA GPU

With the optional `nvidia-ml-py` dependency and a compatible NVIDIA driver, PowerNap can discover:

- Stable GPU identity
- GPU and memory utilization
- Encoder and decoder activity
- Temperature
- Power draw
- Current, minimum, maximum, and default power limits

Power limits are clamped to the range reported by the driver and verified after application.

### AMD GPU

For AMDGPU devices exposing documented sysfs controls, PowerNap can discover:

- GPU identity
- Current performance level
- Available power profile modes
- Current, minimum, and maximum power caps when exposed through hwmon

The current adapter plans documented `low`, `auto`, and `high` performance-level changes. Fan control, overdrive tables, voltage control, and undocumented registers are outside the project scope.

### Intel GPU

Intel GPU support is observation-first. PowerNap does not write Intel GPU controls unless a documented, discoverable, and verifiable interface is implemented for the active driver.

### Other systems

A system without a supported GPU remains usable. A system without CPUFreq can still be inspected, but CPU control is unavailable and `powernap check` reports the missing capability when CPU management is enabled.

## How it works

```text
Capability discovery
        |
        v
CPU, GPU, thermal, workload, and price observations
        |
        v
Immutable system state
        |
        +--> Demand floor
        +--> Price preference
        +--> Thermal safety ceiling
        |
        v
Profile selection
        |
        v
Transition gating and hysteresis
        |
        v
Capability-specific control plan
        |
        v
Apply, read back, verify, record
```

The decision relationship is conceptually:

```text
selected profile = clamp(price preference, demand floor, safety ceiling)
```

This prevents price from being counted several times and ensures workload demand cannot be reduced below the required floor.

## Safety model

PowerNap changes low-level power controls, so safety is part of the architecture rather than an optional feature.

- **Dry-run by default:** the supplied configuration calculates and records simulated plans without writing hardware controls.
- **Capability-driven planning:** values are selected only from discovered interfaces and reported ranges.
- **Thermal ceiling:** warm and hot conditions cap escalation; critical temperature forces Eco immediately.
- **Read-back verification:** a change is marked applied only when the resulting value matches the request.
- **Rollback:** when a required operation fails, previously applied operations in that plan are restored where possible.
- **Baseline restoration:** managed file-backed controls can be restored during a clean shutdown.
- **External-change handling:** the default `yield` policy stops active management after conflicting external changes are detected.
- **No overclocking or undervolting:** PowerNap does not bypass manufacturer limits or firmware thermal protection.
- **No global PCI power sweep:** PowerNap does not blindly enable runtime power management across every PCI device.

Dry-run output must be reviewed on every target computer before physical control is enabled.

## Requirements

- Linux
- Python 3.10 or newer
- `psutil`
- `requests`
- Write access to selected sysfs controls when physical management is enabled
- Optional `nvidia-ml-py` for NVIDIA telemetry and control
- systemd only when using the supplied service unit

## Installation

### Clone and create a virtual environment

```bash
git clone https://github.com/swetoast/PowerNap.git
cd PowerNap
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test,nvidia]'
```


If the system has no NVIDIA GPU, the optional NVIDIA dependency can be omitted:

```bash
python -m pip install -e '.[test]'
```

## Quick start

### 1. Inspect capabilities

```bash
powernap --config powernap.conf capabilities
```

### 2. Validate configuration and environment

```bash
powernap --config powernap.conf check
```

### 3. Generate a read-only plan without network prices

```bash
powernap --config powernap.conf --dry-run once --no-price
```

### 4. Generate a read-only plan with electricity prices

```bash
powernap --config powernap.conf --dry-run once
```

Do not set `dry_run = false` until the capabilities and representative plans have been reviewed on the target hardware.

## Configuration

The included `powernap.conf` is a safe starting point and remains in dry-run mode.

Important settings:

```ini
[general]
dry_run = true
external_change_policy = yield
restore_on_exit = true

[price]
area = SE3
timezone = Europe/Stockholm
provider = elpris_eu
fallback_provider = elprisetjustnu

[thermal]
warm_temp_c = 60
hot_temp_c = 75
critical_temp_c = 90

[cpu]
enabled = true
powercap_enabled = false

[gpu]
nvidia_enabled = true
amdgpu_enabled = true
```

### Protected processes

A comma-separated process list can establish a Responsive minimum profile while matching processes are active:

```ini
[workloads]
protected_processes = ffmpeg, HandBrakeCLI
```

Names are examples only. PowerNap does not hardcode application-specific rules.

### Electricity areas

Supported Swedish areas are `SE1`, `SE2`, `SE3`, and `SE4`.

### External changes

`external_change_policy` accepts:

- `yield`: stop managing after a conflicting external change
- `observe`: accept the external state
- `manage`: continue applying PowerNap decisions

`yield` is the safest default when another power-management service may be active.

## Command reference

### Run the daemon

```bash
powernap --config /etc/powernap/powernap.conf run
```

### Inspect capabilities

```bash
powernap --config powernap.conf capabilities
```

### Validate the environment

```bash
powernap --config powernap.conf check
```

### Evaluate one cycle

```bash
powernap --config powernap.conf --dry-run once
```

### Evaluate without a network request

```bash
powernap --config powernap.conf --dry-run once --no-price
```

### Show current price context

```bash
powernap --config powernap.conf prices
```

### Show latest recorded status

```bash
powernap --config powernap.conf status
```

### Show recent decisions and control events

```bash
powernap --config powernap.conf report
```

## Example decisions

### Idle system during an expensive interval

```text
Demand floor: Eco
Price preference: Eco
Thermal ceiling: Maximum
Selected profile: Eco
```

### Single busy core on a many-core CPU

```text
Average utilization: low
Peak-core utilization: high
Demand floor: Responsive
Selected profile: Responsive
```

Peak-core demand prevents a single-threaded workload from being misclassified as idle.

### Active GPU video workload while electricity is expensive

```text
GPU demand floor: Responsive
Price preference: Eco
Thermal ceiling: Maximum
Selected profile: Responsive
```

Measured workload demand overrides the lower price preference.

### Critical temperature

```text
Thermal ceiling: Eco
Selected profile: Eco
Transition: immediate
```

Critical thermal protection bypasses normal transition holds.

## Running as a systemd service

First install PowerNap and validate it in dry-run mode. Then install the configuration and service:

```bash
sudo install -d -m 0755 /etc/powernap /var/lib/powernap
sudo install -m 0644 powernap.conf /etc/powernap/powernap.conf
sudo install -m 0644 systemd/powernap.service /etc/systemd/system/powernap.service
sudo systemctl daemon-reload
sudo systemctl enable --now powernap.service
```

Inspect service state and logs:

```bash
sudo systemctl status powernap.service
sudo journalctl -u powernap.service -f
```

The service uses systemd readiness and watchdog notifications. Its hardening settings still allow access to the selected sysfs and database paths required for power management.

## Data and privacy

PowerNap runs locally and has no web server or remote-control interface.

It stores local records in SQLite:

- System observation summaries
- Profile decisions
- Control attempts and verification results
- Normalized price records and metadata where implemented

Process protection matches configured executable or process names. It does not inspect document contents or network payloads.

Electricity-price requests are sent only to the configured provider and optional fallback provider.

## Testing

Install test dependencies and run:

```bash
python -m pip install -e '.[test,nvidia]'
pytest
```

The current automated suite covers:

- Configuration parsing and validation
- Inline comments on boolean settings
- Single-core burst detection
- Missing-price behavior
- Critical thermal override
- Transition candidate behavior
- Generic CPUFreq fixture discovery
- Governor mapping
- SQLite cycle persistence

Hardware integration testing must be performed explicitly because automated tests do not write to the host's real power controls.

## Validation status

### Verified in the packaged build

- Python module compilation
- 123 regression tests with measured branch coverage, covering capabilities, configuration, CLI behavior, control ordering, rollback, dry-run isolation, thermal safety and recovery, transitions, electricity prices, database migration, workloads, packaging, and telemetry
- One-shot dry-run execution
- JSON validation for one-shot, capability, and check output
- Final ZIP integrity and content inspection
- Removal of bytecode, caches, and test artifacts from the archive

### Not yet verified

- Physical writes on every CPUFreq, EPP, AMDGPU, NVIDIA, and RAPL implementation
- Long-term service stability on a broad hardware set
- systemd watchdog behavior on every distribution
- Live price-provider behavior in every network environment
- The design target of 85 percent branch coverage. Current measured branch coverage is 83.90 percent

See [docs/VALIDATION.md](docs/VALIDATION.md) for the exact validation record.

## Roadmap

Before a 1.0 release, the project needs:

- Real-hardware validation for each advertised writable adapter
- Wider tests for hourly, quarter-hour, DST, missing, duplicate, overlapping, and out-of-order price intervals
- Cache-corruption recovery and additional price-quality diagnostics
- Expanded AMDGPU and Intel GPU validation
- Physical RAPL application and restoration tests on supported hardware
- Service installation and upgrade tests
- Branch coverage measurement and expansion toward the documented target
- Long-duration daemon and suspend/resume testing
- Distribution-specific installation documentation

## Contributing

Contributions are welcome. Start with [CONTRIBUTING.md](CONTRIBUTING.md), run the regression suite, and keep new hardware controls capability-driven, bounded, and verifiable.

A new hardware adapter should include:

- Documented discovery logic
- Valid range detection
- Dry-run output
- Read-back verification
- Failure handling
- Fixture-based tests
- Real-hardware validation notes

## Security

Do not publish suspected vulnerabilities in a public issue. Follow [SECURITY.md](SECURITY.md) for reporting guidance.

PowerNap can run with privileges capable of changing hardware power settings. Review configuration, service permissions, dependency sources, and dry-run plans before deployment.

## License

PowerNap is licensed under the [GNU General Public License v3.0](LICENSE). You may use, study, modify, and redistribute the software under the terms of that license.

The project repository is [swetoast/PowerNap](https://github.com/swetoast/PowerNap). Issues and source history are maintained there.
