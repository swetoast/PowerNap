# PowerNap

PowerNap is a Linux CPU governor controller that adjusts CPU power behavior based on real-time electricity prices and current CPU load. It is designed for small servers, home labs, and always-on Linux machines where you want lower power use during expensive hours without making the system feel unstable.

PowerNap monitors CPU usage, fetches electricity prices from `elprisetjustnu.se`, stores price and CPU history in SQLite, and changes the active CPU frequency governor when price and load conditions justify it.

## Features

- **Electricity-price-aware CPU scaling**  
  Uses current Swedish electricity price data to avoid unnecessary high-performance CPU behavior during expensive periods.

- **Real-time CPU monitoring**  
  Samples per-core CPU usage with `psutil` and uses smoothed usage for more stable decisions.

- **Governor hysteresis**  
  Uses separate enter/exit thresholds to reduce rapid switching between governors.

- **Cooldown after upscaling**  
  Prevents immediate downscaling after the governor has been raised for performance.

- **Configurable governor logic**  
  Thresholds, smoothing, cooldown, polling interval, retention, logging, and database location are configurable in `powernap.conf`.

- **SQLite history**  
  Stores electricity prices and CPU usage history in local SQLite databases.

- **Price prefetching**  
  Fetches today’s prices and attempts to fetch tomorrow’s prices after midday when available.

- **Resilient API handling**  
  Uses request timeouts, response validation, and exponential backoff with jitter.

- **Clean shutdown**  
  Handles `SIGTERM` and `SIGINT`, commits buffered CPU data, and closes databases cleanly.

- **Systemd-friendly**  
  Includes an example service file with basic hardening options.

## Requirements

- Linux with CPU frequency scaling support exposed under:

  ```text
  /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
  ```

- Python 3.10 or newer recommended
- Python packages:

  ```text
  psutil
  requests
  ```

- Permission to write to CPU governor files
- Network access to fetch electricity prices

## Project Files

```text
powernap.py              Main PowerNap service script
powernap.conf.example    Example configuration file
powernap.service         Example systemd service unit
README.md                This document
```

Runtime files created by PowerNap:

```text
prices.db                Electricity price history
cpu.db                   CPU usage history
```

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/PowerNap.git
cd PowerNap
```

Create and activate a virtual environment:

```bash
python3 -m venv env
source env/bin/activate
```

Install dependencies:

```bash
pip install psutil requests
```

Copy the example configuration:

```bash
cp powernap.conf.example powernap.conf
chmod 600 powernap.conf
```

Run PowerNap manually:

```bash
python3 powernap.py
```

Run debug output:

```bash
python3 powernap.py --debug
```

## Configuration

PowerNap reads `powernap.conf` from the script directory by default. You can override the config path with:

```bash
POWERNAP_CONFIG=/path/to/powernap.conf python3 powernap.py
```

Example configuration:

```ini
[CostConstants]
HIGH_COST = 1.0
MID_COST = 0.5
LOW_COST = 0.1

[AreaCode]
AREA = SE3

[SleepInterval]
INTERVAL = 5

[DataRetention]
ENABLED = true
DAYS = 30

[CommitInterval]
INTERVAL = 5

[UsageCalculation]
METHOD = median

[GovernorLogic]
PERF_ENTER_UTIL = 85
PERF_EXIT_UTIL = 70
PSAVE_ENTER_UTIL = 30
PSAVE_EXIT_UTIL = 40
SMOOTH_FACTOR = 0.30
COOLDOWN_SEC = 15

[Network]
REQUEST_TIMEOUT_SEC = 10

[Storage]
DATABASE_DIR = .

[Logging]
LEVEL = INFO
```

### CostConstants

Electricity price thresholds in SEK/kWh.

```ini
[CostConstants]
HIGH_COST = 1.0
MID_COST = 0.5
LOW_COST = 0.1
```

The values should be ordered like this:

```text
HIGH_COST > MID_COST > LOW_COST
```

### AreaCode

Swedish electricity price area used when fetching prices.

```ini
[AreaCode]
AREA = SE3
```

Common Swedish areas are:

```text
SE1
SE2
SE3
SE4
```

### SleepInterval

How often the main loop runs, in seconds.

```ini
[SleepInterval]
INTERVAL = 5
```

### DataRetention

Controls automatic cleanup of old database rows.

```ini
[DataRetention]
ENABLED = true
DAYS = 30
```

### CommitInterval

How often buffered CPU usage samples are committed to SQLite, in minutes.

```ini
[CommitInterval]
INTERVAL = 5
```

Set to `0` to disable periodic CPU history commits.

### UsageCalculation

Controls how per-core CPU usage is aggregated for runtime decisions.

```ini
[UsageCalculation]
METHOD = median
```

Supported values:

```text
median
average
```

`median` is usually less jumpy. `average` is more sensitive to broad load across many cores.

### GovernorLogic

Controls CPU governor decision behavior.

```ini
[GovernorLogic]
PERF_ENTER_UTIL = 85
PERF_EXIT_UTIL = 70
PSAVE_ENTER_UTIL = 30
PSAVE_EXIT_UTIL = 40
SMOOTH_FACTOR = 0.30
COOLDOWN_SEC = 15
```

- `PERF_ENTER_UTIL`: CPU usage required to enter `performance` when price is low.
- `PERF_EXIT_UTIL`: CPU usage must fall below this before leaving `performance`.
- `PSAVE_ENTER_UTIL`: CPU usage low enough to enter `powersave` when price is high.
- `PSAVE_EXIT_UTIL`: CPU usage must rise above this before leaving `powersave`.
- `SMOOTH_FACTOR`: Exponential moving average factor. Must be greater than `0` and less than or equal to `1`.
- `COOLDOWN_SEC`: Minimum time after an upscale before allowing a downscale.

### Network

```ini
[Network]
REQUEST_TIMEOUT_SEC = 10
```

Timeout for electricity price API requests.

### Storage

```ini
[Storage]
DATABASE_DIR = .
```

Directory where `prices.db` and `cpu.db` are stored. Relative paths are resolved from the script directory.

### Logging

```ini
[Logging]
LEVEL = INFO
```

Common values:

```text
DEBUG
INFO
WARNING
ERROR
```

## Governor Decision Overview

PowerNap chooses between these governors when available:

```text
powersave
conservative
schedutil
performance
```

The rough behavior is:

- High CPU usage and low electricity price can allow `performance`.
- Moderate or high usage with acceptable price can use `schedutil`.
- Moderate usage with low price can use `conservative`.
- Expensive power or low load pushes toward `powersave`.

If a chosen governor is not supported by the system, PowerNap falls back to the closest supported governor based on governor priority.

## Systemd Service

An example `powernap.service` is included.

Recommended install layout:

```bash
sudo useradd --system --home /opt/powernap --shell /usr/sbin/nologin powernap
sudo mkdir -p /opt/powernap
sudo cp powernap.py powernap.conf.example /opt/powernap/
sudo cp /opt/powernap/powernap.conf.example /opt/powernap/powernap.conf
sudo chown -R powernap:powernap /opt/powernap
sudo chmod 700 /opt/powernap
sudo chmod 600 /opt/powernap/powernap.conf
sudo chmod 755 /opt/powernap/powernap.py
```

Install the service:

```bash
sudo cp powernap.service /etc/systemd/system/powernap.service
sudo systemctl daemon-reload
sudo systemctl enable --now powernap.service
```

Check logs:

```bash
journalctl -u powernap.service -f
```

Example service:

```systemd
[Unit]
Description=PowerNap electricity-price-aware CPU governor controller
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=powernap
Group=powernap
WorkingDirectory=/opt/powernap
Environment=POWERNAP_CONFIG=/opt/powernap/powernap.conf
ExecStart=/usr/bin/python3 /opt/powernap/powernap.py
Restart=on-failure
NoNewPrivileges=true
PrivateTmp=true
ProtectHome=true
ProtectSystem=full
ReadWritePaths=/opt/powernap /sys/devices/system/cpu

[Install]
WantedBy=multi-user.target
```

## Permission Notes

PowerNap needs permission to write to files like:

```text
/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
```

Depending on your distro and kernel, this may require one of these approaches:

- Run the service as root.
- Use a tightly scoped helper for governor writes.
- Use a carefully limited sudoers rule.
- Adjust udev/system permissions if appropriate for your system.

Avoid giving broad root access if you do not need it.

## Security Notes

- Keep `powernap.conf` owner-readable only:

  ```bash
  chmod 600 powernap.conf
  ```

- Keep database files private if they are stored in a shared location.
- Run as a dedicated user where possible.
- Use systemd hardening options where possible.
- Avoid storing PowerNap in world-writable directories.

## Debugging

Show database contents:

```bash
python3 powernap.py --debug
```

Check supported governors manually:

```bash
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors
```

Check current governor:

```bash
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
```

Check service logs:

```bash
journalctl -u powernap.service -n 100 --no-pager
```

## Troubleshooting

### No governor files found

Your CPU, kernel, VM, container, or power driver may not expose CPU frequency governor control.

Check:

```bash
ls /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

### Permission denied when setting governor

The service user does not have permission to write to `scaling_governor`.

Check service logs:

```bash
journalctl -u powernap.service -f
```

### No current electricity price found

PowerNap will try to fetch price data for the current date. If the API is unavailable, it backs off and retries later.

Check logs for API errors:

```bash
journalctl -u powernap.service -f
```

### Governor keeps changing too often

Increase smoothing or cooldown:

```ini
[GovernorLogic]
SMOOTH_FACTOR = 0.20
COOLDOWN_SEC = 30
```

You can also widen the hysteresis thresholds:

```ini
PERF_ENTER_UTIL = 90
PERF_EXIT_UTIL = 65
PSAVE_ENTER_UTIL = 25
PSAVE_EXIT_UTIL = 45
```

## License

This project is licensed under the GPL-3.0 License.
