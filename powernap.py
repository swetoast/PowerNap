
#!/usr/bin/env python3
"""
PowerNap - CPU Governor Adjuster based on real-time electricity prices.

This script monitors CPU usage and dynamically adjusts the CPU frequency governor according
to current electricity prices. It uses an open API to fetch hourly or quarter-hourly electricity prices
for the specified region and uses thresholds to decide whether to favor performance or power saving.

Improvements in this version:
 - Structured code into modular functions and classes for readability and maintainability.
 - Replaced per-core sequential CPU usage sampling with a single multi-core sampling for efficiency.
 - Avoids redundant CPU governor changes by tracking the last applied governor.
 - Added robust error handling for configuration loading, API requests, and I/O operations.
 - Supports quarter-hour price data (96 intervals per day) and fetches next day's prices when available.
 - Preserves data logging to SQLite for CPU usage and prices, with data retention as configured.
 - Configuration values are validated with default fallbacks.
"""
import os
import sys
import time
import glob
import sqlite3
import requests
import psutil
import configparser
from datetime import datetime, timedelta
from collections import deque
from statistics import mean, median

# Configuration Loading
def load_config(config_path):
    """Load configuration from the given file path, with defaults for missing values."""
    config = configparser.ConfigParser()
    # Preserve case for options
    config.optionxform = str
    if not config.read(config_path):
        print(f"Warning: Configuration file not found at {config_path}. Using default settings.")
    def get_option(section, option, cast=str, default=None):
        try:
            value = config.get(section, option)
        except (configparser.NoSectionError, configparser.NoOptionError):
            if default is None:
                raise
            return default
        # Remove inline comments for string/boolean values
        if cast is str or cast is bool:
            value = value.partition('#')[0].strip()
        return config.getboolean(section, option) if cast is bool else cast(value)
    defaults = {
        ('CostConstants', 'HIGH_COST'): 1.0,
        ('CostConstants', 'MID_COST'): 0.5,
        ('CostConstants', 'LOW_COST'): 0.1,
        ('AreaCode', 'AREA'): 'SE3',
        ('SleepInterval', 'INTERVAL'): 5,
        ('DataRetention', 'DAYS'): 30,
        ('DataRetention', 'ENABLED'): True,
        ('CommitInterval', 'INTERVAL'): 5,
        ('UsageCalculation', 'METHOD'): 'median',
    }
    cfg = {}
    try:
        cfg['HIGH_COST'] = get_option('CostConstants', 'HIGH_COST', float, defaults[('CostConstants','HIGH_COST')])
        cfg['MID_COST'] = get_option('CostConstants', 'MID_COST', float, defaults[('CostConstants','MID_COST')])
        cfg['LOW_COST'] = get_option('CostConstants', 'LOW_COST', float, defaults[('CostConstants','LOW_COST')])
        cfg['AREA'] = get_option('AreaCode', 'AREA', str, defaults[('AreaCode','AREA')])
        cfg['SLEEP_INTERVAL'] = get_option('SleepInterval', 'INTERVAL', int, defaults[('SleepInterval','INTERVAL')])
        cfg['DATA_RETENTION_DAYS'] = get_option('DataRetention', 'DAYS', int, defaults[('DataRetention','DAYS')])
        cfg['DATA_RETENTION_ENABLED'] = get_option('DataRetention', 'ENABLED', bool, defaults[('DataRetention','ENABLED')])
        cfg['COMMIT_INTERVAL'] = get_option('CommitInterval', 'INTERVAL', int, defaults[('CommitInterval','INTERVAL')])
        method_val = get_option('UsageCalculation', 'METHOD', str, defaults[('UsageCalculation','METHOD')]).lower()
        if method_val not in {'average', 'median'}:
            print(f"Warning: Unknown usage calculation method '{method_val}', defaulting to 'median'.")
            method_val = 'median'
        cfg['USAGE_METHOD'] = method_val
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)
    return cfg

# Database Managers
class DatabaseManager:
    """Base class for managing SQLite database connections and operations."""
    def __init__(self, db_file):
        try:
            self.conn = sqlite3.connect(db_file, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
        except sqlite3.Error as e:
            print(f"Error: Unable to connect to database {db_file}: {e}")
            sys.exit(1)
        self.conn.row_factory = sqlite3.Row
        self._set_pragmas()
    def _set_pragmas(self):
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.conn.execute("PRAGMA temp_store=MEMORY;")
    def execute(self, query, params=None, commit=False):
        cur = self.conn.cursor()
        try:
            if params:
                cur.execute(query, params)
            else:
                cur.execute(query)
            if commit:
                self.conn.commit()
        except sqlite3.Error as e:
            print(f"Database query failed: {e}\nQuery: {query}")
            return []
        return cur.fetchall()
    def executemany(self, query, seq_of_params, commit=False):
        cur = self.conn.cursor()
        try:
            cur.executemany(query, seq_of_params)
            if commit:
                self.conn.commit()
        except sqlite3.Error as e:
            print(f"Database batch query failed: {e}\nQuery: {query}")
            return []
        return cur.fetchall()
    def remove_old_data(self, days):
        """Remove data older than the specified number of days from prices and cpu_usage tables."""
        try:
            with self.conn:
                self.conn.execute("DELETE FROM prices WHERE time_start < datetime('now', ?)", (f'-{days} days',))
                self.conn.execute("DELETE FROM cpu_usage WHERE timestamp < datetime('now', ?)", (f'-{days} days',))
        except sqlite3.Error as e:
            print(f"Warning: Failed to prune old data: {e}")

class PriceManager(DatabaseManager):
    """Manages price data in the database."""
    def __init__(self, db_file):
        super().__init__(db_file)
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS prices (
                    id INTEGER PRIMARY KEY,
                    SEK_per_kWh REAL NOT NULL,
                    time_start TEXT NOT NULL,
                    time_end TEXT NOT NULL,
                    UNIQUE(SEK_per_kWh, time_start, time_end)
                )"""
        )
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_prices_time ON prices(time_start, time_end);")
        self.conn.commit()
    def insert_bulk(self, data_list):
        if not data_list:
            return
        self.executemany(
            "INSERT OR IGNORE INTO prices(SEK_per_kWh, time_start, time_end) VALUES (?,?,?)",
            data_list,
            commit=True
        )
    def insert_data(self, data):
        self.insert_bulk([data])
    def get_current_price(self):
        now_iso = datetime.now().isoformat()
        rows = self.execute(
            "SELECT SEK_per_kWh FROM prices WHERE time_start <= ? AND time_end > ? ORDER BY time_start DESC LIMIT 1",
            (now_iso, now_iso)
        )
        return rows[0]["SEK_per_kWh"] if rows else None
    def has_data_for_date(self, date_str):
        pattern = date_str + '%'
        rows = self.execute("SELECT 1 FROM prices WHERE time_start LIKE ? LIMIT 1", (pattern,))
        return len(rows) > 0

class CPUManager(DatabaseManager):
    """Manages CPU usage data in the database and in-memory buffer."""
    def __init__(self, db_file, core_count):
        super().__init__(db_file)
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS cpu_usage (
                    id INTEGER PRIMARY KEY,
                    timestamp TEXT,
                    cpu_cores INTEGER NOT NULL,
                    cpu_core_id INTEGER NOT NULL,
                    cpu_usage REAL NOT NULL,
                    cpu_governor TEXT NOT NULL
                )"""
        )
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_cpu_time ON cpu_usage(timestamp);")
        self.conn.commit()
        self.core_count = core_count
        self.cpu_data = {i: deque(maxlen=900) for i in range(core_count)}
    def insert_temp(self, data_tuple):
        _, _, core_id, _, _ = data_tuple
        if core_id in self.cpu_data:
            self.cpu_data[core_id].append(data_tuple)
    def commit_data(self, current_governor):
        now_iso = datetime.now().isoformat()
        batch = []
        for cid, dq in self.cpu_data.items():
            if dq:
                usage_vals = [entry[3] for entry in dq]
                usage_median = median(usage_vals)
                batch.append((now_iso, self.core_count, cid, usage_median, current_governor))
        if batch:
            self.executemany(
                "INSERT INTO cpu_usage(timestamp, cpu_cores, cpu_core_id, cpu_usage, cpu_governor) VALUES (?,?,?,?,?)",
                batch,
                commit=True
            )
            self.cpu_data = {i: deque(maxlen=900) for i in range(self.core_count)}
            print("CPU usage data committed to database.")
    def show(self):
        rows = self.execute("SELECT * FROM cpu_usage")
        for row in rows:
            print(dict(row))
    def get_usage(self, core_id, method='median'):
        data_points = [entry[3] for entry in self.cpu_data.get(core_id, [])]
        if not data_points:
            return None
        return mean(data_points) if method == 'average' else median(data_points)

class DataFetcher:
    @staticmethod
    def fetch(area_code, date_str):
        url = f'https://www.elprisetjustnu.se/api/v1/prices/{date_str}_{area_code}.json'
        try:
            response = requests.get(url, timeout=10)
        except requests.RequestException as e:
            print(f"Error: Failed to fetch price data from API: {e}")
            return None
        if response.status_code != 200:
            print(f"Warning: API request returned status {response.status_code}")
            return None
        try:
            data = response.json()
        except ValueError as e:
            print(f"Error: Invalid JSON data received: {e}")
            return None
        return data

class CPUMonitor:
    @staticmethod
    def get_current_governor():
        for gov_file in glob.glob('/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor'):
            try:
                with open(gov_file, 'r') as f:
                    return f.read().strip()
            except IOError:
                continue
        return None
    @staticmethod
    def choose_governor(cpu_usage, price, thresholds):
        high_cost, mid_cost, low_cost = thresholds
        if price is None:
            return 'powersave'
        if cpu_usage > 85 and price < low_cost:
            return 'performance'
        if 70 <= cpu_usage <= 85 and price < mid_cost:
            return 'schedutil'
        if cpu_usage < 30 and price > high_cost:
            return 'powersave'
        if 30 <= cpu_usage < 70 and price < mid_cost:
            return 'conservative'
        return 'powersave'

def set_cpu_governor(governor):
    success = True
    for gov_file in glob.glob('/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor'):
        try:
            with open(gov_file, 'w') as f:
                f.write(governor)
        except IOError as e:
            success = False
            print(f"Failed to set governor to '{governor}': {e}")
            break
    return success

def main():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    cfg = load_config(os.path.join(script_dir, 'powernap.conf'))
    # Initialize database managers
    price_mgr = PriceManager(os.path.join(script_dir, 'prices.db'))
    core_count = psutil.cpu_count(logical=True) or 1
    cpu_mgr = CPUManager(os.path.join(script_dir, 'cpu.db'), core_count)
    # Debug mode
    if '-debug' in sys.argv:
        price_rows = price_mgr.execute("SELECT * FROM prices")
        print("Price data:")
        for row in price_rows:
            print(dict(row))
        print("CPU data:")
        cpu_mgr.show()
        return
    current_governor = CPUMonitor.get_current_governor() or 'unknown'
    thresholds = (cfg['HIGH_COST'], cfg['MID_COST'], cfg['LOW_COST'])
    last_cleanup = datetime.now()
    last_commit_minute = None

    while True:
        now = datetime.now()
        current_price = price_mgr.get_current_price()
        if current_price is None:
            today_str = now.strftime('%Y/%m-%d')
            fetched = DataFetcher.fetch(cfg['AREA'], today_str)
            if fetched:
                today_data = [(item['SEK_per_kWh'], item['time_start'], item['time_end']) for item in fetched]
                price_mgr.insert_bulk(today_data)
                print("Today's price data inserted.")
            if now.hour >= 13:
                tomorrow_date = now.date() + timedelta(days=1)
                tomorrow_str = tomorrow_date.strftime('%Y/%m-%d')
                if not price_mgr.has_data_for_date(str(tomorrow_date)):
                    fetched_tomorrow = DataFetcher.fetch(cfg['AREA'], tomorrow_str)
                    if fetched_tomorrow:
                        tomorrow_data = [(item['SEK_per_kWh'], item['time_start'], item['time_end']) for item in fetched_tomorrow]
                        price_mgr.insert_bulk(tomorrow_data)
                        print("Tomorrow's price data inserted.")
            current_price = price_mgr.get_current_price()
        # CPU usage sampling
        timestamp = now.isoformat()
        try:
            usage_list = psutil.cpu_percent(interval=1, percpu=True)
        except Exception as e:
            print(f"Error reading CPU usage: {e}")
            usage_list = [0.0] * core_count
        for cid, usage in enumerate(usage_list):
            cpu_mgr.insert_temp((timestamp, core_count, cid, usage, current_governor))
        smoothed_usages = []
        for cid in range(core_count):
            val = cpu_mgr.get_usage(cid, cfg['USAGE_METHOD'])
            if val is not None:
                smoothed_usages.append(val)
        overall_usage = max(smoothed_usages) if smoothed_usages else 0.0
        new_governor = CPUMonitor.choose_governor(overall_usage, current_price, thresholds)
        if new_governor != current_governor:
            if set_cpu_governor(new_governor):
                print(f"Governor changed from {current_governor} to {new_governor}")
                current_governor = new_governor
        if cfg['COMMIT_INTERVAL'] > 0:
            minute = now.minute
            if minute % cfg['COMMIT_INTERVAL'] == 0:
                if last_commit_minute != minute:
                    cpu_mgr.commit_data(current_governor)
                    last_commit_minute = minute
        if cfg['DATA_RETENTION_ENABLED']:
            if (datetime.now() - last_cleanup).days >= cfg['DATA_RETENTION_DAYS']:
                price_mgr.remove_old_data(cfg['DATA_RETENTION_DAYS'])
                cpu_mgr.remove_old_data(cfg['DATA_RETENTION_DAYS'])
                last_cleanup = datetime.now()
                print("Old data pruned.")
        time.sleep(cfg['SLEEP_INTERVAL'])

if __name__ == "__main__":
    main()
