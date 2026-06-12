# PowerNap

PowerNap is a formulaic, policy-gated Linux CPU governor controller that adjusts CPU power behaviour based on real-time electricity prices, current CPU load, workload shape, price lookahead, and thermal state.

It is designed for small servers, home labs, and always-on Linux machines where you want lower power use during expensive electricity hours without making the system feel unstable or sluggish.

PowerNap monitors CPU usage, fetches electricity prices from `elprisetjustnu.se`, stores price and CPU history in SQLite, and changes the active CPU frequency governor when the decision engine determines that price, workload, and safety conditions justify it.

PowerNap v2 uses:

- CLI execution
- systemd logs
- SQLite history
- PrettyTable presentation reports via an explicit command

## Features

- **Electricity-price-aware CPU scaling**  
  Uses Swedish electricity price data to avoid unnecessary high-performance CPU behaviour during expensive periods.

- **Formulaic decision engine**  
  Uses calculated scores instead of simple fixed `if price > x and cpu < y` logic.

- **Policy-gated behaviour**  
  Uses a policy matrix as a safe baseline before applying formulaic score adjustments.

- **Workload classification**  
  Classifies runtime behaviour into workload types such as idle, light, balanced, burst, sustained heavy, and I/O-bound.

- **Price rank classification**  
  Compares the current electricity price against the day’s price range instead of relying only on fixed SEK/kWh thresholds.

- **Price lookahead**  
  Looks ahead over upcoming price data so cheap periods before expensive windows can be used more intelligently.

- **Thermal safety layer**  
  CPU temperature can reduce aggressiveness, and critical temperature overrides all other decisions.

- **Transition gating**  
  Uses minimum hold time, upscale cooldown, and score-change margin to avoid twitchy governor switching.

- **Real-time CPU monitoring**  
  Samples per-core CPU usage with `psutil` and smooths usage for stable decisions.

- **SQLite history**  
  Stores electricity price history, CPU usage history, and governor change events in local SQLite databases.

- **Price prefetching**  
  Fetches today’s prices and attempts to fetch tomorrow’s prices after midday when available.

- **Resilient API handling**  
  Uses request timeouts, response validation, and exponential backoff with jitter.

- **Clean shutdown**  
  Handles `SIGTERM` and `SIGINT`, commits buffered CPU data, and closes databases cleanly.

- **PrettyTable report command**  
  Generates readable presentation output using a dedicated `--report` command.

- **Systemd-friendly**  
  Includes an example systemd service file with basic hardening options.

## Requirements

- Linux with CPU frequency scaling support exposed under:

  ```text
  /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
  ```

- Python 3.10 or newer recommended
- Python packages:

  ```text
  requests
  psutil
  prettytable
  ```

- Permission to write to CPU governor files
- Network access to fetch electricity prices

## Project Files

```text
powernap.py              Main PowerNap service script
powernap.conf.example    Example configuration file
requirements.txt         Python dependencies
powernap.service         Example systemd service unit
install.sh               Basic install helper
README.md                This document
```

Runtime files created by PowerNap:

```text
prices.db                Electricity price history
cpu.db                   CPU usage history and governor events
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
pip install -r requirements.txt
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

Run in dry-run mode without writing governors:

```bash
python3 powernap.py --dry-run
```

Generate a presentation report:

```bash
python3 powernap.py --report
```

Show raw diagnostic output:

```bash
python3 powernap.py --debug
```

Validate config and environment:

```bash
python3 powernap.py --check-config
```

## Runtime Model

PowerNap is meant to run as a long-running daemon loop.

The daemon loop:

```text
fetches prices
samples CPU usage
calculates workload state
calculates price state
calculates thermal state
builds a decision
applies transition gating
changes governor if needed
writes history to SQLite
sleeps until the next cycle
```

Reports are separate commands. The daemon does not start a report thread.

The intended model is:

```text
PowerNap daemon process
  writes SQLite history

PowerNap report command
  reads SQLite history
  prints PrettyTable output
  exits
```

SQLite WAL mode is used, so this read/write pattern is suitable for a daemon writing history while a separate report command reads the latest committed data.

## Configuration

PowerNap reads `powernap.conf` from the script directory by default. You can override the config path with:

```bash
POWERNAP_CONFIG=/path/to/powernap.conf python3 powernap.py
```

Example configuration:

```ini
[AreaCode]
AREA = SE3

[Runtime]
SLEEP_INTERVAL = 5
COMMIT_INTERVAL_MINUTES = 5
USAGE_METHOD = median
DRY_RUN = false

[Storage]
DATABASE_DIR = .

[Logging]
LEVEL = INFO

[DataRetention]
ENABLED = true
DAYS = 30

[Network]
REQUEST_TIMEOUT_SEC = 10

[Metrics]
SMOOTH_FACTOR = 0.30
HISTORY_SAMPLES = 24
HIGH_LOAD_THRESHOLD = 70
BURST_CPU_THRESHOLD = 75
IDLE_CPU_THRESHOLD = 15
IOWAIT_THRESHOLD = 25

[PriceStrategy]
CHEAP_RANK_MAX = 0.25
EXPENSIVE_RANK_MIN = 0.80
LOOKAHEAD_HOURS = 3
LOOKAHEAD_DELTA = 0.20

[Formula]
CPU_WEIGHT = 0.55
URGENCY_WEIGHT = 0.20
CHEAP_NOW_WEIGHT = 0.15
LOOKAHEAD_WEIGHT = 0.10
THERMAL_WEIGHT = 0.25

[Thermal]
ENABLED = true
WARM_TEMP = 70
HOT_TEMP = 80
CRITICAL_TEMP = 90

[Transition]
MIN_HOLD_SEC = 60
UPSCALE_COOLDOWN_SEC = 15
SCORE_MARGIN = 7.5
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

### Runtime

Controls the main daemon loop and runtime behaviour.

```ini
[Runtime]
SLEEP_INTERVAL = 5
COMMIT_INTERVAL_MINUTES = 5
USAGE_METHOD = median
DRY_RUN = false
```

- `SLEEP_INTERVAL`: how often the main loop runs, in seconds.
- `COMMIT_INTERVAL_MINUTES`: how often buffered CPU samples are committed to SQLite.
- `USAGE_METHOD`: how per-core CPU usage is aggregated for runtime decisions.
- `DRY_RUN`: calculates decisions without writing to governor files.

Supported usage methods:

```text
median
average
max
```

`median` is usually less jumpy. `average` is more sensitive to broad load across cores. `max` reacts more strongly to the busiest core.

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

### DataRetention

Controls automatic cleanup of old database rows.

```ini
[DataRetention]
ENABLED = true
DAYS = 30
```

### Network

```ini
[Network]
REQUEST_TIMEOUT_SEC = 10
```

Timeout for electricity price API requests.

### Metrics

Controls CPU smoothing and workload classification thresholds.

```ini
[Metrics]
SMOOTH_FACTOR = 0.30
HISTORY_SAMPLES = 24
HIGH_LOAD_THRESHOLD = 70
BURST_CPU_THRESHOLD = 75
IDLE_CPU_THRESHOLD = 15
IOWAIT_THRESHOLD = 25
```

- `SMOOTH_FACTOR`: exponential moving average factor. Must be greater than `0` and less than or equal to `1`.
- `HISTORY_SAMPLES`: number of smoothed samples used for sustained-load detection.
- `HIGH_LOAD_THRESHOLD`: CPU percentage considered high for sustained-load detection.
- `BURST_CPU_THRESHOLD`: CPU percentage that can classify a short load as burst.
- `IDLE_CPU_THRESHOLD`: CPU percentage considered idle when load average is also low.
- `IOWAIT_THRESHOLD`: I/O wait percentage that can classify the workload as I/O-bound.

### PriceStrategy

Controls daily price rank classification and lookahead behaviour.

```ini
[PriceStrategy]
CHEAP_RANK_MAX = 0.25
EXPENSIVE_RANK_MIN = 0.80
LOOKAHEAD_HOURS = 3
LOOKAHEAD_DELTA = 0.20
```

PowerNap compares the current price against the day’s price range:

```text
0.0 = cheapest point in the day
1.0 = most expensive point in the day
```

- `CHEAP_RANK_MAX`: rank at or below this is treated as cheap.
- `EXPENSIVE_RANK_MIN`: rank at or above this is treated as expensive.
- `LOOKAHEAD_HOURS`: number of future hours used for price lookahead.
- `LOOKAHEAD_DELTA`: rank difference considered meaningful for rising/falling price trends.

### Formula

Controls the formulaic aggression score.

```ini
[Formula]
CPU_WEIGHT = 0.55
URGENCY_WEIGHT = 0.20
CHEAP_NOW_WEIGHT = 0.15
LOOKAHEAD_WEIGHT = 0.10
THERMAL_WEIGHT = 0.25
```

The score is calculated as:

```text
score =
    load_score      * CPU_WEIGHT
  + urgency_score   * URGENCY_WEIGHT
  + cheap_now_score * CHEAP_NOW_WEIGHT
  + lookahead_score * LOOKAHEAD_WEIGHT
  - thermal_penalty * THERMAL_WEIGHT
```

The score does not blindly decide everything. It adjusts the policy baseline while safety overrides and transition rules still apply.

### Thermal

Controls thermal safety behaviour.

```ini
[Thermal]
ENABLED = true
WARM_TEMP = 70
HOT_TEMP = 80
CRITICAL_TEMP = 90
```

- `WARM_TEMP`: begins reducing aggressiveness.
- `HOT_TEMP`: reduces aggressiveness more strongly.
- `CRITICAL_TEMP`: overrides everything and forces `powersave`.

### Transition

Controls stability around governor changes.

```ini
[Transition]
MIN_HOLD_SEC = 60
UPSCALE_COOLDOWN_SEC = 15
SCORE_MARGIN = 7.5
```

- `MIN_HOLD_SEC`: minimum time before allowing a normal downshift.
- `UPSCALE_COOLDOWN_SEC`: minimum time after an upscale before allowing a downscale.
- `SCORE_MARGIN`: minimum score movement required before accepting a governor change.

## Decision Model

PowerNap uses a layered decision model:

```text
1. Safety overrides
2. Metrics collection
3. Workload classification
4. Price classification
5. Policy matrix baseline
6. Formulaic aggression adjustment
7. Transition gating
```

### Workload Classes

```text
idle
light
balanced
burst
sustained_heavy
io_bound
```

### Price Classes

```text
cheap
normal
expensive
unknown
```

### Thermal Classes

```text
normal
warm
hot
critical
unknown
```

### Policy Matrix Baseline

```text
                 cheap          normal         expensive       unknown
idle             powersave      powersave      powersave       powersave
light            conservative   powersave      powersave       powersave
balanced         schedutil      conservative   powersave       conservative
burst            schedutil      schedutil      conservative    conservative
sustained_heavy  performance    schedutil      schedutil       schedutil
io_bound         conservative   conservative   powersave       conservative
```

The formula score can adjust the baseline by one level, but critical thermal state overrides everything.

## Governor Decision Overview

PowerNap chooses between these governors when available:

```text
powersave
conservative
schedutil
performance
```

The rough behaviour is:

- Idle or light load tends toward `powersave`.
- Balanced load can use `conservative` or `schedutil` depending on price rank.
- Burst load can use `schedutil` when price conditions are not hostile.
- Sustained heavy load can use `performance` when electricity is cheap.
- Expensive power suppresses non-urgent workloads.
- High temperature reduces aggressiveness.
- Critical temperature forces `powersave`.

If a chosen governor is not supported by the system, PowerNap falls back to the closest supported governor based on governor priority.

## Commands

Run the daemon:

```bash
python3 powernap.py
```

Run in dry-run mode:

```bash
python3 powernap.py --dry-run
```

Generate a PrettyTable report:

```bash
python3 powernap.py --report
```

Show raw diagnostic output:

```bash
python3 powernap.py --debug
```

Validate configuration and environment:

```bash
python3 powernap.py --check-config
```

Use a specific config file:

```bash
python3 powernap.py --config /path/to/powernap.conf
```

## PrettyTable Reports

PrettyTable is used by the explicit report command:

```bash
python3 powernap.py --report
```

The report reads SQLite history and prints presentation-friendly tables for:

```text
latest prices
governor events
CPU usage samples
```

This is a separate process from the daemon. It does not run as a thread inside the daemon.

## Systemd Service

An example `powernap.service` is included.

Recommended install layout:

```bash
sudo mkdir -p /opt/powernap
sudo cp powernap.py powernap.conf.example requirements.txt /opt/powernap/
sudo cp /opt/powernap/powernap.conf.example /opt/powernap/powernap.conf
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
Description=PowerNap formulaic CPU governor controller
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=/opt/powernap
Environment=POWERNAP_CONFIG=/opt/powernap/powernap.conf
ExecStart=/usr/bin/python3 /opt/powernap/powernap.py
Restart=on-failure
RestartSec=5
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
- Run as a dedicated service user where possible.
- Use systemd hardening options where possible.
- Avoid storing PowerNap in world-writable directories.
- Do not expose PowerNap files or SQLite databases through a web server.

## Debugging

Show raw database contents:

```bash
python3 powernap.py --debug
```

Show presentation output:

```bash
python3 powernap.py --report
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

Increase transition stability:

```ini
[Transition]
MIN_HOLD_SEC = 120
UPSCALE_COOLDOWN_SEC = 30
SCORE_MARGIN = 10
```

Increase smoothing:

```ini
[Metrics]
SMOOTH_FACTOR = 0.20
```

### PowerNap reacts too slowly

Reduce transition stability:

```ini
[Transition]
MIN_HOLD_SEC = 30
UPSCALE_COOLDOWN_SEC = 10
SCORE_MARGIN = 5
```

Increase smoothing responsiveness:

```ini
[Metrics]
SMOOTH_FACTOR = 0.40
```

### Reports show no data

The daemon may not have committed history yet, or it may be using a different database directory.

Check:

```bash
python3 powernap.py --check-config
```

Also confirm:

```ini
[Storage]
DATABASE_DIR = .
```

## License

This project is licensed under the GPL-3.0 License.
