# PowerNap: Your Eco-Friendly CPU Monitor

Welcome to PowerNap! This is a Python-based tool that keeps an eye on your CPU usage and smartly adjusts the CPU governor to optimize power consumption. It's like having a personal assistant for your CPU, but one that's eco-friendly!

## Features

- **Real-time CPU Usage Monitoring**: PowerNap keeps track of your CPU usage in real time.
- **Smart Governor Adjustment**: PowerNap adjusts the CPU governor based on the current CPU usage and power cost, helping you save power.
- **Energy Consumption Logging**: PowerNap logs energy consumption and CO2 emissions data, allowing you to understand your energy usage better.
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
4. Install the required Python packages. The required packages are `psutil`, `cpufreq`, `eco2ai`, `nordpool`, `configparser`, `sqlite3`, `threading`, `sklearn`:
    ```bash
    pip install psutil cpufreq eco2ai nordpool configparser sqlite3 threading sklearn
    ```

## Usage

1. Update the `powernap.conf` configuration file with your desired settings.
2. Run the script:
    ```bash
    python powernap.py
    ```

## Configuration

The `powernap.conf` file contains the following configuration options:

- `area`: The area for which you want to get the power cost.
- `sleep_time`: The time interval (in seconds) between each monitoring cycle.
- `low`, `mid`, `high`: Thresholds for power cost categories.
- `currency`: The currency for power cost.

## Systemd Service

To run PowerNap as a systemd service, create a service file `powernap.service`:

```systemd
[Unit]
Description=PowerNap Service
After=network.target

[Service]
ExecStart=/path/to/your/python/env/bin/python /path/to/your/script/powernap.py
Restart=always

[Install]
WantedBy=multi-user.target
```
## License
This project is licensed under the GPL-3.0 License.
