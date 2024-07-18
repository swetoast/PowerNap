# PowerNap: Your Eco-Friendly CPU Monitor

Welcome to PowerNap! This is a Python-based tool that keeps an eye on your CPU usage and smartly adjusts the CPU governor to optimize power consumption based on real-time electricity prices. It's like having a personal assistant for your CPU, but one that's eco-friendly!

## Features

- **Real-time CPU Usage Monitoring**: PowerNap keeps track of your CPU usage in real time.
- **Smart Governor Adjustment**: PowerNap adjusts the CPU governor based on the current CPU usage and electricity prices, helping you save power.
- **Real-time Electricity Price Monitoring**: PowerNap fetches real-time electricity prices from an API and adjusts the CPU governor accordingly.
- **Data Management**: PowerNap stores CPU usage and electricity price data in SQLite databases and provides options for data retention.

## Installation

1. Clone this repository to your local machine:
    ```bash
    git clone https://github.com/yourusername/PowerNap.git
    ```
2. Navigate to the cloned repository:
    ```bash
    cd PowerNap
    ```
3. Create a Python virtual environment and activate it:
    ```bash
    python3 -m venv env
    source env/bin/activate
    ```
4. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```
## Configuration

The `powernap.conf` file contains several configuration options:

- `Database`: This section contains several configuration options related to the database and the main loop of the script.
    - `AREA`: This is the area code for which to fetch electricity prices. The value should be a string that represents the area code.
    - `SLEEP_INTERVAL`: This is the interval (in seconds) at which the script should sleep between iterations of the main loop. The value should be an integer that represents the number of seconds.
    - `DAYS`: This is the number of days to retain data in the database. The value should be an integer that represents the number of days.
    - `ENABLED`: This is a boolean value that indicates whether data retention is enabled. If it's set to `True`, the script will remove data older than the number of days specified in `DAYS` from the database.
    - `COMMIT_INTERVAL`: This is the interval (in minutes) at which data should be committed to the database. The value should be an integer that represents the number of minutes.
    - `METHOD`: This specifies the method for calculating CPU usage. The value can be either `average` or `median`.

## Rules Configuration

The `rules.json` file contains a set of rules that PowerNap uses to decide which CPU governor to use based on the current CPU usage and electricity prices. Each rule is a JSON object with the following properties:

- `usage_lower_bound`: The lower bound of the CPU usage for this rule.
- `usage_upper_bound`: The upper bound of the CPU usage for this rule (only used if `usage_comparison` is `between`).
- `usage_comparison`: The comparison operator for the CPU usage. Can be `higher_then`, `lower_then`, `between`, or `default`.
- `power_cost_value`: The value of the electricity price for this rule.
- `power_cost_comparison`: The comparison operator for the power cost. Can be `lower_then`, `higher_then`, or `default`.
- `governor`: The CPU governor to use if the rule applies.

For detailed guidance on rule creation, please refer to the  [rules.md](rules.md) document.

## Systemd Service

To run PowerNap as a systemd service, create a service file `powernap.service`:

```systemd
[Unit]
Description=PowerNap Service
After=network.target

[Service]
ExecStart=/path/to/your/python/env/bin/python -u /path/to/your/script/powernap.py
Restart=always

[Install]
WantedBy=multi-user.target
```
## License
This project is licensed under the GPL-3.0 License.
