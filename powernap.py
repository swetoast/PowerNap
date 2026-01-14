#!/usr/bin/env python3
"""
PowerNap - Enhanced CPU Governor Adjuster based on real-time electricity prices.

This script monitors CPU usage and dynamically adjusts the CPU frequency governor according 
to current electricity prices, with advanced strategies to reduce power cost while maintaining 
performance stability.

New features in this refactored version:
- **Hysteresis Logic:** Separate up/down CPU usage thresholds prevent rapid governor flapping. 
  The CPU must drop significantly below an "up" threshold before downshifting, and rise sufficiently 
  above a "down" threshold before upshifting.
- **Cooldown Period:** After scaling up (to a higher-performance governor), it waits a short period 
  before allowing a scale-down. This avoids bouncing governors due to short transient changes.
- **Smoothed CPU Measurement:** Uses an exponential moving average of CPU usage to stabilize short-term 
  spikes and dips when making decisions.
- **Robust Governor Selection:** Chooses among 'performance', 'schedutil', 'conservative', and 'powersave' 
  based on smoothed usage trends and electricity price (high/mid/low cost thresholds). Global (system-wide) 
  governor is set, as modern systems typically apply frequency policy per processor package.
- **Resilient API Handling:** Implements exponential backoff with jitter for price data fetches to handle 
  network issues gracefully without spamming the API. All API calls enforce HTTPS with certificate verification.
- **Security Enhancements:** Encourages least-privilege operation. Config and database files are locked 
  down to owner-only access. The code is designed to be run as a dedicated user with minimal system 
  permissions (see Deployment notes below).

Deployment Security Tips:
- **Least Privilege:** Create a dedicated user (e.g., "powernap") to run this script. Ensure the 
  configuration (powernap.conf) and database files are owned by this user and have permissions 600 
  (read/write by owner only).
- **Systemd Hardening (Example):**
    - User=powernap (run as non-root user)
    - NoNewPrivileges=true, PrivateTmp=true, ProtectSystem=full, ProtectHome=true
    - ReadWritePaths=/path/to/powernap /sys/devices/system/cpu/ (allow writing only to required paths)
    - CapabilityBoundingSet=CAP_SYS_ADMIN (to permit governor changes without full root, if supported)
- **Sudo Alternative:** If not using systemd, configure /etc/sudoers to allow the powernap user to execute 
  only the necessary commands (e.g., writing to scaling_governor) as root, rather than running the entire 
  script as root.
"""
import os
import sys
import time
import glob
import sqlite3
import requests
import psutil
import configparser
import random
from datetime import datetime, timedelta
from collections import deque
from statistics import mean, median

# Define hysteresis and smoothing constants
PERF_ENTER_UTIL = 85   # CPU usage % threshold to enter performance mode
PERF_EXIT_UTIL  = 70   # CPU usage % threshold to exit performance mode (hysteresis lower than enter)
PSAVE_ENTER_UTIL = 30  # CPU usage % threshold to enter powersave mode
PSAVE_EXIT_UTIL  = 40  # CPU usage % threshold to exit powersave mode (hysteresis higher than enter)
SMOOTH_FACTOR   = 0.3  # Smoothing factor for exponential moving average (0 < alpha <= 1)
COOLDOWN_SEC    = 15   # Minimum seconds to wait after an upscale before allowing a downscale
GOVERNOR_PRIORITY = { 'powersave': 0, 'conservative': 1, 'schedutil': 2, 'performance': 3 }

# Configuration Loading and Validation
def load_config(config_path):
    """Load configuration from file, applying defaults and validations."""
    config = configparser.ConfigParser()
    config.optionxform = str  # preserve case
    if not config.read(config_path):
        print(f"Warning: Config file not found at {config_path}, using defaults.")
    def get_opt(section, option, cast=str, default=None):
        try:
            value = config.get(section, option)
        except (configparser.NoSectionError, configparser.NoOptionError):
            if default is None:
                raise
            return default
        # Strip inline comments for string/bool types
        if cast in (str, bool):
            value = value.split('#')[0].strip()
        return config.getboolean(section, option) if cast is bool else cast(value)
    cfg = {}
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
    try:
        cfg['HIGH_COST'] = get_opt('CostConstants', 'HIGH_COST', float, defaults[('CostConstants','HIGH_COST')])
        cfg['MID_COST']  = get_opt('CostConstants', 'MID_COST', float, defaults[('CostConstants','MID_COST')])
        cfg['LOW_COST']  = get_opt('CostConstants', 'LOW_COST', float, defaults[('CostConstants','LOW_COST')])
        if not (cfg['HIGH_COST'] > cfg['MID_COST'] > cfg['LOW_COST']):
            print(f"Warning: Cost thresholds not in strict HIGH>MID>LOW order (HIGH={cfg['HIGH_COST']}, MID={cfg['MID_COST']}, LOW={cfg['LOW_COST']}).")
        cfg['AREA'] = get_opt('AreaCode', 'AREA', str, defaults[('AreaCode','AREA')])
        cfg['SLEEP_INTERVAL'] = get_opt('SleepInterval', 'INTERVAL', int, defaults[('SleepInterval','INTERVAL')])
        cfg['DATA_RETENTION_DAYS'] = get_opt('DataRetention', 'DAYS', int, defaults[('DataRetention','DAYS')])
        cfg['DATA_RETENTION_ENABLED'] = get_opt('DataRetention', 'ENABLED', bool, defaults[('DataRetention','ENABLED')])
        cfg['COMMIT_INTERVAL'] = get_opt('CommitInterval', 'INTERVAL', int, defaults[('CommitInterval','INTERVAL')])
        method_val = get_opt('UsageCalculation', 'METHOD', str, defaults[('UsageCalculation','METHOD')]).lower()
        if method_val not in ('average', 'median'):
            print(f"Warning: Unknown usage calculation method '{method_val}', defaulting to 'median'.")
            method_val = 'median'
        cfg['USAGE_METHOD'] = method_val
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)
    return cfg

# Database Managers
class DatabaseManager:
    def __init__(self, db_file):
        try:
            self.conn = sqlite3.connect(db_file, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
        except sqlite3.Error as e:
            print(f"Error: Cannot open database {db_file}: {e}")
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
            cur.execute(query, params or ())
            if commit:
                self.conn.commit()
        except sqlite3.Error as e:
            print(f"Database error: {e}\n Query: {query}")
            return []
        return cur.fetchall()
    def executemany(self, query, seq_of_params, commit=False):
        cur = self.conn.cursor()
        try:
            cur.executemany(query, seq_of_params)
            if commit:
                self.conn.commit()
        except sqlite3.Error as e:
            print(f"Database batch error: {e}\n Query: {query}")
            return []
        return cur.fetchall()
    def remove_old_data(self, days):
        """Delete records older than 'days' days from relevant tables."""
        try:
            with self.conn:
                self.conn.execute("DELETE FROM prices WHERE time_start < datetime('now', ?)", (f'-{days} days',))
                self.conn.execute("DELETE FROM cpu_usage WHERE timestamp < datetime('now', ?)", (f'-{days} days',))
        except sqlite3.Error as e:
            print(f"Warning: failed to prune old data: {e}")

class PriceManager(DatabaseManager):
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
    def get_current_price(self):
        now_iso = datetime.now().isoformat()
        rows = self.execute(
            "SELECT SEK_per_kWh FROM prices WHERE time_start <= ? AND time_end > ? ORDER BY time_start DESC LIMIT 1",
            (now_iso, now_iso)
        )
        return rows[0]["SEK_per_kWh"] if rows else None
    def has_data_for_date(self, date_str):
        # Check if any price entry exists for the given date (YYYY-MM-DD prefix).
        pattern = date_str + '%'
        rows = self.execute("SELECT 1 FROM prices WHERE time_start LIKE ? LIMIT 1", (pattern,))
        return len(rows) > 0

class CPUManager(DatabaseManager):
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
        # Buffer for recent usage data (in-memory) for each core
        self.cpu_data = {i: deque(maxlen=900) for i in range(core_count)}
    def insert_temp(self, data_tuple):
        # Append a raw usage record for a core to the in-memory buffer
        _, _, core_id, _, _ = data_tuple
        if core_id in self.cpu_data:
            self.cpu_data[core_id].append(data_tuple)
    def commit_data(self, current_governor):
        # Persist median usage for each core in the current interval to the database
        now_iso = datetime.now().isoformat()
        batch = []
        for cid, dq in self.cpu_data.items():
            if dq:
                usage_vals = [entry[3] for entry in dq]
                core_usage = median(usage_vals)  # median of collected samples
                batch.append((now_iso, self.core_count, cid, core_usage, current_governor))
        if batch:
            self.executemany(
                "INSERT INTO cpu_usage(timestamp, cpu_cores, cpu_core_id, cpu_usage, cpu_governor) VALUES (?,?,?,?,?)",
                batch,
                commit=True
            )
            # Reset buffers after committing
            self.cpu_data = {i: deque(maxlen=900) for i in range(self.core_count)}
            print("[INFO] CPU usage data committed to database.")
    def get_usage(self, core_id, method='median'):
        data_points = [entry[3] for entry in self.cpu_data.get(core_id, [])]
        if not data_points:
            return None
        return mean(data_points) if method == 'average' else median(data_points)

class DataFetcher:
    @staticmethod
    def fetch(area_code, date_str):
        # Fetch price data for given date (format YYYY/MM-DD) and area code.
        url = f'https://www.elprisetjustnu.se/api/v1/prices/{date_str}_{area_code}.json'
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()  # Raise an exception for HTTP errors (4xx/5xx)
        except requests.RequestException as e:
            print(f"[ERROR] Price API request failed: {e}")
            return None
        try:
            data = r.json()
        except ValueError as e:
            print(f"[ERROR] Failed to parse API response: {e}")
            return None
        # Basic validation of response structure
        if isinstance(data, list) and data and all(isinstance(item.get('SEK_per_kWh'), (int, float)) for item in data):
            return data
        print(f"[ERROR] Unexpected API response format: {data}")
        return None

class CPUMonitor:
    @staticmethod
    def get_current_governor():
        # Read the current governor of the first CPU (assuming uniform setting across CPUs)
        for gov_file in glob.glob('/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor'):
            try:
                with open(gov_file, 'r') as f:
                    return f.read().strip()
            except IOError:
                continue
        return None
    @staticmethod
    def choose_governor(current_governor, usage, cost, thresholds):
        high_cost, mid_cost, low_cost = thresholds
        # Determine desired governor based on smoothed usage and price.
        if cost is None:
            # If no price info, default to powersave (safe mode).
            desired = 'powersave'
        elif usage is None:
            # If usage data is unavailable, keep current or default to powersave.
            desired = current_governor or 'powersave'
        else:
            # Base decision (no hysteresis yet)
            if usage >= PERF_ENTER_UTIL and cost < low_cost:
                desired = 'performance'
            elif usage >= 70 and cost < mid_cost:
                # Moderately high usage, price is not too high
                desired = 'schedutil'
            elif usage <= PSAVE_ENTER_UTIL and cost > high_cost:
                desired = 'powersave'
            elif usage <= 50 and cost >= mid_cost:
                # Low usage and at least mid-level cost
                desired = 'powersave'
            elif usage < 70 and cost < mid_cost:
                # Moderate usage and low cost
                desired = 'conservative'
            else:
                # Default to a conservative powersave approach if no condition matches
                desired = 'powersave'
        # Apply hysteresis: require extra margin to change state if currently at an extreme.
        if current_governor == 'performance' and desired != 'performance':
            if usage is not None and usage >= PERF_EXIT_UTIL:
                desired = 'performance'  # Stay in performance until usage drops below PERF_EXIT_UTIL
        if current_governor == 'powersave' and desired != 'powersave':
            if usage is not None and usage <= PSAVE_EXIT_UTIL:
                desired = 'powersave'  # Stay in powersave until usage rises above PSAVE_EXIT_UTIL
        return desired

def set_cpu_governor(governor):
    """Attempt to set the CPU frequency governor for all CPU cores."""
    success = True
    for gov_file in glob.glob('/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor'):
        try:
            with open(gov_file, 'w') as f:
                f.write(governor)
        except IOError as e:
            success = False
            print(f"[ERROR] Failed to set governor to '{governor}': {e}")
            break
    return success

def main():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    cfg = load_config(os.path.join(script_dir, 'powernap.conf'))

    # Apply secure file permissions to config and database files (owner RW only)
    try:
        os.chmod(os.path.join(script_dir, 'powernap.conf'), 0o600)
    except Exception:
        pass
    price_db_path = os.path.join(script_dir, 'prices.db')
    cpu_db_path   = os.path.join(script_dir, 'cpu.db')

    price_mgr = PriceManager(price_db_path)
    cpu_mgr   = CPUManager(cpu_db_path, psutil.cpu_count(logical=True) or 1)
    # Lock down database files
    try:
        os.chmod(price_db_path, 0o600)
        os.chmod(cpu_db_path, 0o600)
    except Exception:
        pass

    # Debug mode: output database contents and exit
    if '-debug' in sys.argv:
        prices = price_mgr.execute("SELECT * FROM prices")
        print("Prices database:")
        for row in prices:
            print(dict(row))
        print("CPU usage database:")
        cpu_rows = cpu_mgr.execute("SELECT * FROM cpu_usage")
        for row in cpu_rows:
            print(dict(row))
        return

    current_governor = CPUMonitor.get_current_governor() or 'unknown'
    thresholds = (cfg['HIGH_COST'], cfg['MID_COST'], cfg['LOW_COST'])

    # Variables for API backoff and smoothing
    next_price_fetch_time = datetime.now()
    fetch_retry_count = 0
    smoothed_usage = None
    last_upscale_time = None

    last_cleanup = datetime.now()
    last_commit_minute = None

    while True:
        now = datetime.now()
        now_iso = now.isoformat()

        # Retrieve current price or fetch if not available
        current_price = price_mgr.get_current_price()
        if current_price is None:
            if now < next_price_fetch_time:
                # Not yet time to retry fetching prices
                pass
            else:
                date_str = now.strftime('%Y/%m-%d')
                fetched = DataFetcher.fetch(cfg['AREA'], date_str)
                if fetched:
                    today_data = [(item['SEK_per_kWh'], item['time_start'], item['time_end']) for item in fetched]
                    price_mgr.insert_bulk(today_data)
                    print("[INFO] Inserted today's electricity prices.")
                    # Reset backoff on success
                    fetch_retry_count = 0
                    next_price_fetch_time = now
                else:
                    # Apply exponential backoff for next retry
                    fetch_retry_count += 1
                    base_delay = min(60 * (2 ** fetch_retry_count), 3600)  # Cap backoff at 1 hour
                    jitter = random.uniform(0, base_delay * 0.1)
                    next_price_fetch_time = now + timedelta(seconds=(base_delay + jitter))
                    print(f"[WARN] Price fetch failed, will retry in ~{int(base_delay)} seconds.")
            # Attempt to pre-fetch tomorrow's prices after midday
            if now.hour >= 13:
                tomorrow_date = now.date() + timedelta(days=1)
                tomorrow_str = tomorrow_date.strftime('%Y/%m-%d')
                if not price_mgr.has_data_for_date(str(tomorrow_date)):
                    fetched_t = DataFetcher.fetch(cfg['AREA'], tomorrow_str)
                    if fetched_t:
                        tdata = [(item['SEK_per_kWh'], item['time_start'], item['time_end']) for item in fetched_t]
                        price_mgr.insert_bulk(tdata)
                        print("[INFO] Inserted tomorrow's electricity prices.")
        # Update current_price after any fetch attempt
        current_price = price_mgr.get_current_price()

        # Sample CPU usage (all cores) and update smoothing
        try:
            usage_list = psutil.cpu_percent(interval=1, percpu=True)
        except Exception as e:
            print(f"[ERROR] psutil.cpu_percent failed: {e}")
            usage_list = [0.0] * (psutil.cpu_count(logical=True) or 1)
        overall_usage = max(usage_list) if usage_list else 0.0
        # Initialize or update exponential moving average of CPU usage
        if smoothed_usage is None:
            smoothed_usage = overall_usage
        else:
            smoothed_usage = SMOOTH_FACTOR * overall_usage + (1 - SMOOTH_FACTOR) * smoothed_usage

        # Buffer per-core usage data for logging (not used in decision directly)
        for cid, u in enumerate(usage_list):
            cpu_mgr.insert_temp((now_iso, cpu_mgr.core_count, cid, u, current_governor))

        # Decide on appropriate governor using smoothed usage and current price
        new_governor = CPUMonitor.choose_governor(current_governor, smoothed_usage, current_price, thresholds)
        # Cooldown logic: prevent quick downscale after an upscale
        if (current_governor in GOVERNOR_PRIORITY) and (new_governor in GOVERNOR_PRIORITY):
            if GOVERNOR_PRIORITY[new_governor] < GOVERNOR_PRIORITY[current_governor]:
                # Proposed a lower-performance governor than current (downscale)
                if last_upscale_time and (datetime.now() - last_upscale_time).total_seconds() < COOLDOWN_SEC:
                    new_governor = current_governor  # hold current governor during cooldown period
            elif GOVERNOR_PRIORITY[new_governor] > GOVERNOR_PRIORITY[current_governor]:
                # Upscaling: record time of this upscale
                last_upscale_time = datetime.now()

        # Apply the governor change if needed
        if new_governor and new_governor != current_governor:
            if set_cpu_governor(new_governor):
                print(f"[INFO] Governor changed from {current_governor} to {new_governor}")
                current_governor = new_governor

        # Periodically commit usage data to the database
        if cfg['COMMIT_INTERVAL'] > 0:
            minute = now.minute
            if minute % cfg['COMMIT_INTERVAL'] == 0:
                if last_commit_minute != minute:
                    cpu_mgr.commit_data(current_governor)
                    last_commit_minute = minute

        # Periodically clean up old data from databases
        if cfg['DATA_RETENTION_ENABLED']:
            if (datetime.now() - last_cleanup).days >= cfg['DATA_RETENTION_DAYS']:
                price_mgr.remove_old_data(cfg['DATA_RETENTION_DAYS'])
                cpu_mgr.remove_old_data(cfg['DATA_RETENTION_DAYS'])
                last_cleanup = datetime.now()
                print(f"[INFO] Purged data older than {cfg['DATA_RETENTION_DAYS']} days.")

        # Sleep until next monitoring cycle
        time.sleep(cfg['SLEEP_INTERVAL'])

if __name__ == "__main__":
    main()
