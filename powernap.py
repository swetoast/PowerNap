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
        self.conn = sqlite3.connect(db_file, detect_types=sqlite3.PARSE_DECLTYPES)
        self.conn.row_factory = sqlite3.Row
        self._set_pragmas()

    def _set_pragmas(self):
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.conn.execute("PRAGMA temp_store=MEMORY;")

    def execute(self, query, params=None, commit=False):
        cur = self.conn.execute(query, params or ())
        if commit:
            self.conn.commit()
        return cur.fetchall()

    def executemany(self, query, seq_of_params, commit=False):
        self.conn.executemany(query, seq_of_params)
        if commit:
            self.conn.commit()

    def remove_old_data(self, days):
        with self.conn:
            for table, col in (("prices", "time_start"), ("cpu_usage", "timestamp")):
                self.conn.execute(f"DELETE FROM {table} WHERE {col} < datetime('now', ?)", (f'-{days} days',))

class PriceManager(DatabaseManager):
    def __init__(self, db_file):
        super().__init__(db_file)
        self.conn.execute("""CREATE TABLE IF NOT EXISTS prices (
            id INTEGER PRIMARY KEY,
            SEK_per_kWh REAL NOT NULL,
            time_start TEXT NOT NULL,
            time_end TEXT NOT NULL,
            UNIQUE(SEK_per_kWh, time_start, time_end))""")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_prices_time ON prices(time_start, time_end)")
        self.conn.commit()

    def insert_bulk(self, data_list):
        self.executemany(
            "INSERT OR IGNORE INTO prices(SEK_per_kWh,time_start,time_end) VALUES(?,?,?)",
            data_list, commit=True
        )

    def insert_data(self, data):
        self.insert_bulk([data])

    def show(self):
        rows = self.execute("SELECT * FROM prices")
        self._pretty_print(["ID", "SEK_per_kWh", "Time Start", "Time End"], rows)

    def get_current_price(self):
        now = datetime.now().isoformat()
        rows = self.execute("""SELECT SEK_per_kWh FROM prices
                                WHERE time_start <= ? AND time_end > ?
                                ORDER BY id DESC LIMIT 1""", (now, now))
        return rows[0]["SEK_per_kWh"] if rows else None

    def _pretty_print(self, headers, rows):
        table = PrettyTable()
        table.field_names = headers
        for row in rows:
            table.add_row([row[h] for h in headers])
        print(table)

class CPUManager(DatabaseManager):
    def __init__(self, db_file):
        super().__init__(db_file)
        self.conn.execute("""CREATE TABLE IF NOT EXISTS cpu_usage (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            cpu_cores INTEGER NOT NULL,
            cpu_core_id INTEGER NOT NULL,
            cpu_usage REAL NOT NULL,
            cpu_governor TEXT NOT NULL)""")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_cpu_time ON cpu_usage(timestamp)")
        self.conn.commit()
        self.cpu_data = {i: deque(maxlen=900) for i in range(psutil.cpu_count(logical=False))}

    def insert_temp(self, data):
        self.cpu_data[data[2]].append(data)

    def commit_data(self, current_governor):
        now = datetime.now().isoformat()
        batch = [
            (now, psutil.cpu_count(logical=False), cid, median([d[3] for d in data]), current_governor)
            for cid, data in self.cpu_data.items() if data
        ]
        if batch:
            self.executemany("""INSERT INTO cpu_usage(timestamp,cpu_cores,cpu_core_id,cpu_usage,cpu_governor)
                                VALUES(?,?,?,?,?)""", batch, commit=True)
            self.cpu_data = {i: deque(maxlen=900) for i in self.cpu_data}
            print("CPU data committed.")

    def show(self):
        rows = self.execute("SELECT * FROM cpu_usage")
        PriceManager._pretty_print(self, ["ID", "Timestamp", "CPU Cores", "CPU Core ID", "CPU Usage", "CPU Governor"], rows)

    def get_usage(self, cpu_core_id, method):
        data = [d[3] for d in self.cpu_data[cpu_core_id]]
        if not data:
            return None
        return mean(data) if method == 'average' else median(data)

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
            try:
                with open(path) as f:
                    return f.read().strip()
            except IOError:
                pass
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
            with open(path, 'w') as f:
                f.write(governor)
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
        current_price = price_mgr.get_current_price()

        if current_price is None:
            fetched = DataFetcher.fetch(AREA, datetime.now().strftime('%Y/%m-%d'))
            if fetched:
                price_mgr.insert_bulk([
                    (item['SEK_per_kWh'], item['time_start'], item['time_end'])
                    for item in fetched
                ])
                print("Price data inserted.")
            current_price = price_mgr.get_current_price()

        current_governor = cpu_mgr.execute(
            "SELECT cpu_governor FROM cpu_usage ORDER BY id DESC LIMIT 1"
        )
        current_governor = current_governor[0]["cpu_governor"] if current_governor else CPUMonitor.get_cpu_governor()

        cpu_cores = psutil.cpu_count(logical=False)
        for i in range(cpu_cores):
            cpu_percent, _ = CPUMonitor.get_cpu_info()
            cpu_mgr.insert_temp((now_iso, cpu_cores, i, cpu_percent, current_governor))
            usage = cpu_mgr.get_usage(i, USAGE_CALCULATION_METHOD)
            if usage is not None:
                set_cpu_governor(
                    CPUMonitor.choose_governor(usage, current_price)
                )

        if datetime.now().minute % COMMIT_INTERVAL == 0:
            cpu_mgr.commit_data(current_governor)

        if DATA_RETENTION_ENABLED and (datetime.now() - last_cleanup).days >= DATA_RETENTION_DAYS:
            price_mgr.remove_old_data(DATA_RETENTION_DAYS)
            cpu_mgr.remove_old_data(DATA_RETENTION_DAYS)
            last_cleanup = datetime.now()

        time.sleep(SLEEP_INTERVAL)

if __name__ == "__main__":
    main()
