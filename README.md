# PowerNap: Your Eco-Friendly CPU Monitor

Welcome to PowerNap! This is a Python-based tool that keeps an eye on your CPU usage and smartly adjusts the CPU governor to optimize power consumption. It's like having a personal assistant for your CPU, but one that's eco-friendly!

## Features

- **Real-time CPU Usage Monitoring**: PowerNap keeps track of your CPU usage in real time.
- **Smart Governor Adjustment**: PowerNap adjusts the CPU governor based on the current CPU usage and power cost, helping you save power.
- **Machine Learning Model**: PowerNap uses a machine learning model to predict the optimal governor setting. It's like having a crystal ball for your CPU!

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

The `powernap.conf` file contains the following configuration options:

- `CostConstants`: This section defines the cost thresholds for high, mid, and low electricity prices.
    ```ini
    [CostConstants]
    HIGH_COST = 1.0
    MID_COST = 0.8
    LOW_COST = 0.5
    ```
- `AreaCode`: This section specifies the area code for which to fetch electricity prices.
    ```ini
    [AreaCode]
    AREA = SE3
    ```
- `SleepInterval`: This section specifies the interval (in seconds) at which the script should sleep between iterations of the main loop.
    ```ini
    [SleepInterval]
    INTERVAL = 60
    ```
- `DataRetention`: This section configures data retention settings. `DAYS` is the number of days to retain data, and `ENABLED` is a boolean indicating whether data retention is enabled.
    ```ini
    [DataRetention]
    DAYS = 1
    ENABLED = True
    ```
- `CommitInterval`: This section specifies the interval (in minutes) at which data should be committed to the database.
    ```ini
    [CommitInterval]
    INTERVAL = 5
    ```
- `UsageCalculation`: This section specifies the method for calculating CPU usage. The `METHOD` can be set to either 'average' or 'median'.
    ```ini
    [UsageCalculation]
    METHOD = median
    ```

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
