import os, sys, time, glob, sqlite3, requests, psutil, platform, configparser
from datetime import datetime
from prettytable import PrettyTable
from collections import deque
from statistics import mean, median

script_dir = os.path.dirname(os.path.realpath(__file__))
config = configparser.ConfigParser()
config.read(os.path.join(script_dir, 'powernap.conf'))

get_cfg = lambda section, key, cast=str: cast(config.get(section, key))
DATABASE_PRICES = os.path.join(script_dir, "prices.db")
DATABASE_CPU = os.path.join(script_dir, "cpu.db")

HIGH_COST = get_cfg('CostConstants', 'HIGH_COST', float)
MID_COST = get_cfg('CostConstants', 'MID_COST', float)
LOW_COST = get_cfg('CostConstants', 'LOW_COST', float)
AREA = get_cfg('AreaCode', 'AREA')
SLEEP_INTERVAL = get_cfg('SleepInterval', 'INTERVAL', int)
DATA_RETENTION_DAYS = get_cfg('DataRetention', 'DAYS', int)
DATA_RETENTION_ENABLED = get_cfg('DataRetention', 'ENABLED', config.getboolean)
COMMIT_INTERVAL = get_cfg('CommitInterval', 'INTERVAL', int)
USAGE_CALCULATION_METHOD = get_cfg('UsageCalculation', 'METHOD').split('#')[0].strip()

class DatabaseManager:
    def __init__(self, db_file):
        self.conn = sqlite3.connect(db_file)

    def execute(self, query, params=None):
        cur = self.conn.cursor()
        cur.execute(query, params or ())
        return cur.fetchall()

    def commit(self):
        self.conn.commit()

    def remove_old_data(self, days):
        for table, col in (("prices", "time_start"), ("cpu_usage", "timestamp")):
            self.execute(f"DELETE FROM {table} WHERE {col} < datetime('now', '-{days} days')")
        self.commit()
        print(f"Data older than {days} days removed.")

class PriceManager(DatabaseManager):
    def __init__(self, db_file):
        super().__init__(db_file)
        self.execute("""CREATE TABLE IF NOT EXISTS prices (
            id INTEGER PRIMARY KEY,
            SEK_per_kWh REAL NOT NULL,
            time_start TEXT NOT NULL,
            time_end TEXT NOT NULL,
            UNIQUE(SEK_per_kWh, time_start, time_end))""")
        self.commit()

    def insert_data(self, data):
        try:
            self.execute("INSERT INTO prices(SEK_per_kWh,time_start,time_end) VALUES(?,?,?)", data)
            self.commit()
        except sqlite3.IntegrityError:
            pass

    def show(self):
        rows = self.execute("SELECT * FROM prices")
        self._pretty_print(["ID", "SEK_per_kWh", "Time Start", "Time End"], rows)

    def get_current_price(self):
        now = datetime.now().isoformat()
        rows = self.execute("SELECT SEK_per_kWh FROM prices WHERE time_start <= ? AND time_end > ? ORDER BY id DESC LIMIT 1", (now, now))
        return rows[0][0] if rows else None

    def _pretty_print(self, headers, rows):
        table = PrettyTable()
        table.field_names = headers
        for row in rows:
            table.add_row(row)
        print(table)

class CPUManager(DatabaseManager):
    def __init__(self, db_file):
        super().__init__(db_file)
        self.execute("""CREATE TABLE IF NOT EXISTS cpu_usage (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            cpu_cores INTEGER NOT NULL,
            cpu_core_id INTEGER NOT NULL,
            cpu_usage REAL NOT NULL,
            cpu_governor TEXT NOT NULL)""")
        self.commit()
        self.cpu_data = {i: deque(maxlen=900) for i in range(psutil.cpu_count(logical=False))}

    def insert_temp(self, data):
        self.cpu_data[data[2]].append(data)

    def commit_data(self):
        cur = self.conn.cursor()
        for core_id, deque_data in self.cpu_data.items():
            if deque_data:
                median_usage = median(d[3] for d in deque_data)
                cur.execute("INSERT INTO cpu_usage(timestamp,cpu_cores,cpu_core_id,cpu_usage,cpu_governor) VALUES(?,?,?,?,?)",
                            (datetime.now().isoformat(), psutil.cpu_count(logical=False), core_id, median_usage, CPUMonitor.get_cpu_governor()))
        self.commit()
        self.cpu_data = {i: deque(maxlen=900) for i in self.cpu_data}
        print("CPU data inserted successfully.")

    def show(self):
        rows = self.execute("SELECT * FROM cpu_usage")
        PriceManager._pretty_print(self, ["ID", "Timestamp", "CPU Cores", "CPU Core ID", "CPU Usage", "CPU Governor"], rows)

    def get_current_governor(self):
        rows = self.execute("SELECT cpu_governor FROM cpu_usage ORDER BY id DESC LIMIT 1")
        return rows[0][0] if rows else None

    def get_usage(self, cpu_core_id, method):
        data = [d[3] for d in self.cpu_data[cpu_core_id]]
        if not data: return None
        return mean(data) if method == 'average' else median(data) if method == 'median' else None

class DataFetcher:
    @staticmethod
    def fetch(area, date_today):
        url = f'https://www.elprisetjustnu.se/api/v1/prices/{date_today}_{area}.json'
        r = requests.get(url, timeout=10)
        return r.json() if r.status_code == 200 else None

class CPUMonitor:
    @staticmethod
    def get_cpu_info():
        return psutil.cpu_percent(interval=1), CPUMonitor.get_cpu_governor()

    @staticmethod
    def get_cpu_governor():
        for path in glob.glob('/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor'):
            with open(path) as f:
                return f.read().strip()
        return None

    @staticmethod
    def choose_governor(usage, cost):
        if cost is None: return 'powersave'
        if usage > 85 and cost < LOW_COST: return 'performance'
        if 70 <= usage <= 85 and cost < MID_COST: return 'schedutil'
        if usage < 30 and cost > HIGH_COST: return 'powersave'
        if 30 <= usage < 70 and cost < MID_COST: return 'conservative'
        return 'powersave'

def set_cpu_governor(governor):
    for path in glob.glob('/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor'):
        try:
            with open(path, 'w') as f: f.write(governor)
        except IOError as e:
            print(f"Failed to set governor: {e}")
            return False
    return True

def main():
    price_mgr, cpu_mgr = PriceManager(DATABASE_PRICES), CPUManager(DATABASE_CPU)
    if '-debug' in sys.argv:
        price_mgr.show(); cpu_mgr.show(); return

    last_cleanup = datetime.now()
    while True:
        now_iso = datetime.now().isoformat()
        if not price_mgr.execute("SELECT 1 FROM prices WHERE time_start <= ? AND time_end > ?", (now_iso, now_iso)):
            for item in DataFetcher.fetch(AREA, datetime.now().strftime('%Y/%m-%d')) or []:
                price_mgr.insert_data((item['SEK_per_kWh'], item['time_start'], item['time_end']))
            print("Price data inserted.")

        cpu_cores = psutil.cpu_count(logical=False)
        for i in range(cpu_cores):
            cpu_percent, _ = CPUMonitor.get_cpu_info()
            cpu_mgr.insert_temp((now_iso, cpu_cores, i, cpu_percent, cpu_mgr.get_current_governor()))
            usage = cpu_mgr.get_usage(i, USAGE_CALCULATION_METHOD)
            if usage is not None:
                set_cpu_governor(CPUMonitor.choose_governor(usage, price_mgr.get_current_price()))

        if datetime.now().minute % COMMIT_INTERVAL == 0: cpu_mgr.commit_data()
        if DATA_RETENTION_ENABLED and (datetime.now() - last_cleanup).days >= DATA_RETENTION_DAYS:
            price_mgr.remove_old_data(DATA_RETENTION_DAYS); cpu_mgr.remove_old_data(DATA_RETENTION_DAYS)
            last_cleanup = datetime.now()

        time.sleep(SLEEP_INTERVAL)

if __name__ == "__main__":
    main()
