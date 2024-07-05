import os
import requests
import sqlite3
import sys
import time
import psutil
from datetime import datetime
from prettytable import PrettyTable
import platform
import configparser

# Load the configuration file
config = configparser.ConfigParser()

# Get the directory of the current script
script_dir = os.path.dirname(os.path.realpath(__file__))

# Read the configuration file
config.read(os.path.join(script_dir, 'powernap.conf'))

# Define the paths to the databases
DATABASE_PRICES = os.path.join(script_dir, "prices.db")
DATABASE_CPU = os.path.join(script_dir, "cpu.db")

# Load cost constants from the configuration file
HIGH_COST = config.getfloat('CostConstants', 'HIGH_COST')
MID_COST = config.getfloat('CostConstants', 'MID_COST')
LOW_COST = config.getfloat('CostConstants', 'LOW_COST')

# Load area code from the configuration file
AREA = config.get('AreaCode', 'AREA')

# Load sleep interval from the configuration file
SLEEP_INTERVAL = config.getint('SleepInterval', 'INTERVAL')

# Get the CPU clock speeds
cpu_freq = psutil.cpu_freq()
LOW_CLOCK_SPEED = cpu_freq.min
HIGH_CLOCK_SPEED = cpu_freq.max

class DatabaseManager:
    def __init__(self, db_file):
        self.conn = self.create_connection(db_file)

    def create_connection(self, db_file):

        conn = None
        try:
            conn = sqlite3.connect(db_file)
        except sqlite3.Error as e:
            print(e)
        return conn

    def execute_query(self, query, data=None):
        """ Execute a query """
        cur = self.conn.cursor()
        if data:
            cur.execute(query, data)
            self.conn.commit()
            return cur.lastrowid
        else:
            cur.execute(query)
            return cur.fetchall()

class PriceManager(DatabaseManager):
    def __init__(self, db_file):
        super().__init__(db_file)
        self.create_table()

    def create_table(self):

        query = """CREATE TABLE IF NOT EXISTS prices (
                        id INTEGER PRIMARY KEY,
                        SEK_per_kWh REAL NOT NULL,
                        time_start TEXT NOT NULL,
                        time_end TEXT NOT NULL,
                        UNIQUE(SEK_per_kWh, time_start, time_end));"""
        self.execute_query(query)

    def insert_data(self, data):

        query = ''' INSERT INTO prices(SEK_per_kWh,time_start,time_end)
                    VALUES(?,?,?) '''
        try:
            return self.execute_query(query, data)
        except sqlite3.IntegrityError:
            # Ignore the exception and continue
            pass

    def read_and_present_data(self):

        query = "SELECT * FROM prices"
        rows = self.execute_query(query)

        # Create a PrettyTable instance
        table = PrettyTable()

        # Specify the Column Names while initializing the Table
        table.field_names = ["ID", "SEK_per_kWh", "Time Start", "Time End"]

        # Add rows
        for row in rows:
            table.add_row(row)

        # Print the table
        print(table)

    def get_current_price(self):
        current_hour = datetime.now().isoformat()
        query = f"SELECT SEK_per_kWh FROM prices WHERE time_start <= '{current_hour}' AND time_end > '{current_hour}' ORDER BY id DESC LIMIT 1"
        rows = self.execute_query(query)
        return rows[0][0] if rows else None

class CPUManager(DatabaseManager):
    def __init__(self, db_file):
        super().__init__(db_file)
        self.create_table()

    def create_table(self):

        query = """CREATE TABLE IF NOT EXISTS cpu_usage (
                        id INTEGER PRIMARY KEY,
                        timestamp TEXT,
                        cpu_cores INTEGER NOT NULL,
                        cpu_core_id INTEGER NOT NULL,
                        cpu_usage REAL NOT NULL,
                        cpu_governor TEXT NOT NULL,
                        cpu_temp REAL NOT NULL);"""
        self.execute_query(query)

    def insert_data(self, data):

        query = ''' INSERT INTO cpu_usage(timestamp, cpu_cores,cpu_core_id,cpu_usage,cpu_governor,cpu_temp)
                    VALUES(?,?,?,?,?,?) '''
        return self.execute_query(query, data)

    def read_and_present_data(self):
        """ Query all rows in the cpu_usage table and present them in a nice way """
        query = "SELECT * FROM cpu_usage"
        rows = self.execute_query(query)

        # Create a PrettyTable instance
        table = PrettyTable()

        # Specify the Column Names while initializing the Table
        table.field_names = ["ID", "Timestamp", "CPU Cores", "CPU Core ID", "CPU Usage", "CPU Governor", "CPU Temp"]

        # Add rows
        for row in rows:
            table.add_row(row)

        # Print the table
        print(table)

    def get_current_governor(self):
        query = "SELECT cpu_governor FROM cpu_usage ORDER BY id DESC LIMIT 1"
        rows = self.execute_query(query)
        return rows[0][0] if rows else None

class DataFetcher:
    @staticmethod
    def fetch_data_from_api(area, date_today):

        url = f'https://www.elprisetjustnu.se/api/v1/prices/{date_today}_{area}.json'
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to retrieve data. Status code: {response.status_code}")
            return None

class CPUMonitor:
    @staticmethod
    def get_cpu_usage():
        """Get the current CPU usage as a percentage for each core."""
        return psutil.cpu_percent(interval=1, percpu=True)

    @staticmethod
    def get_cpu_cores():

        return psutil.cpu_count()

    @staticmethod
    def get_cpu_info(core_id):

        governor_cmd = f"cat /sys/devices/system/cpu/cpu{core_id}/cpufreq/scaling_governor"
        temp_cmd = f"cat /sys/class/thermal/thermal_zone0/temp"
        try:
            governor = os.popen(governor_cmd).read().strip()
            if governor in ['userspace', 'schedutil']:
                return None, None
            temp = os.popen(temp_cmd).read().strip()
            temp = float(temp) / 1000
            return governor, temp
        except Exception as e:
            print(f"Error: {e}")
            return None, None

    @staticmethod
    def choose_governor(usage, power_cost, temp):
        clock_speed = psutil.cpu_freq().current
        if power_cost is None:
            return 'powersave'

        if usage > 75 and power_cost < LOW_COST and temp < max_temp:
            return 'performance'
        elif usage < 25 and power_cost > HIGH_COST and temp < max_temp:
            return 'powersave'
        elif 25 <= usage <= 75 and MID_COST <= power_cost <= HIGH_COST and temp < max_temp:
            return 'conservative'
        elif clock_speed > HIGH_CLOCK_SPEED and temp < max_temp:
            return 'performance'
        elif clock_speed < LOW_CLOCK_SPEED and temp > max_temp:
            return 'powersave'
        else:
            return 'ondemand'

def main():
    price_manager = PriceManager(DATABASE_PRICES)
    cpu_manager = CPUManager(DATABASE_CPU)

    current_price = price_manager.get_current_price()
    current_governor = cpu_manager.get_current_governor()

    print(f"Current price: {current_price}")
    print(f"Current governor: {current_governor}")

    if '-debug' in sys.argv:
        print("Price data:")
        price_manager.read_and_present_data()
        print("\nCPU data:")
        cpu_manager.read_and_present_data()
        sys.exit()  # Exit the script

    while True:
        area = AREA
        date_today = datetime.now().strftime('%Y/%m-%d')

        # Check if data for today already exists in the database
        query = f"SELECT * FROM prices WHERE time_start LIKE '{date_today}%'"
        if not price_manager.execute_query(query):
            # If not, fetch data from API and insert into database
            data_prices = DataFetcher.fetch_data_from_api(area, date_today)
            if data_prices is not None:
                for item in data_prices:
                    SEK_per_kWh = item['SEK_per_kWh']
                    time_start = item['time_start']
                    time_end = item['time_end']
                    data_item = (SEK_per_kWh, time_start, time_end)
                    price_manager.insert_data(data_item)
            print("Price data inserted successfully.")
        else:
            print("Price data for today already exists.")

        cpu_cores = CPUMonitor.get_cpu_cores()
        cpu_usages = CPUMonitor.get_cpu_usage()
        for i in range(cpu_cores):
            cpu_governor, cpu_temp = CPUMonitor.get_cpu_info(i)
            if cpu_governor is None and cpu_temp is None:
                continue
            timestamp = datetime.now().isoformat()
            cpu_usage = cpu_usages[i]
            cpu_governor = CPUMonitor.choose_governor(cpu_usage, current_price, cpu_temp)
            data_cpu = (timestamp, cpu_cores, i, cpu_usage, cpu_governor, cpu_temp)
            cpu_manager.insert_data(data_cpu)
        print("CPU data inserted successfully.")
        time.sleep(SLEEP_INTERVAL)

if __name__ == "__main__":
    main()
