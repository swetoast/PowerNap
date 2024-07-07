import os
import requests
import sqlite3
import sys
import time
import psutil
import glob
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

# Load data retention settings from the configuration file
DATA_RETENTION_DAYS = config.getint('DataRetention', 'DAYS')
DATA_RETENTION_ENABLED = config.getboolean('DataRetention', 'ENABLED')

# Load max temperature from the configuration file
MAX_TEMP = config.getfloat('MaxTemp', 'MAX_TEMP')

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

    def remove_old_data(self, days):
        query = f"DELETE FROM prices WHERE time_start < datetime('now', '-{days} days')"
        self.execute_query(query)
        query = f"DELETE FROM cpu_usage WHERE timestamp < datetime('now', '-{days} days')"
        self.execute_query(query)
        print(f"Data older than {days} days removed.")

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
    def get_cpu_temp():
        temp_files = glob.glob('/sys/class/thermal/thermal_zone*/temp')
        for temp_file in temp_files:
            with open(temp_file, 'r') as file:
                temp = int(file.read()) / 1000.0
                return temp
        print("Could not find temperature file.")
        return None

    @staticmethod
    def get_cpu_governor():
        governor_files = glob.glob('/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor')
        for governor_file in governor_files:
            with open(governor_file, 'r') as file:
                governor = file.read().strip()
                return governor
        print("Could not find governor file.")
        return None

    @staticmethod
    def get_cpu_info():
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_freq = psutil.cpu_freq().current
        cpu_temp = CPUMonitor.get_cpu_temp()
        cpu_governor = CPUMonitor.get_cpu_governor()
        return cpu_percent, cpu_freq, cpu_temp, cpu_governor

    @staticmethod
    def choose_governor(usage, power_cost, temp):
        clock_speed = psutil.cpu_freq().current
        if power_cost is None:
            return 'powersave'
        if usage > 85 and power_cost < LOW_COST and temp < MAX_TEMP:
            return 'performance'
        elif 70 <= usage <= 85 and power_cost < MID_COST and temp < MAX_TEMP:
            return 'schedutil'
        elif usage < 30 and power_cost > HIGH_COST and temp < MAX_TEMP:
            return 'powersave'
        elif 30 <= usage < 70 and power_cost < MID_COST and temp < MAX_TEMP:
            return 'conservative'
        elif clock_speed > HIGH_CLOCK_SPEED and temp < MAX_TEMP:
            return 'performance'
        elif clock_speed < LOW_CLOCK_SPEED and temp > MAX_TEMP:
            return 'powersave'

def set_cpu_governor(governor):
    governor_files = glob.glob('/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor')
    for governor_file in governor_files:
        try:
            with open(governor_file, 'w') as file:
                file.write(governor)
        except IOError as e:
            print(f"Failed to set governor: {e}")
            return False
    return True

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

    last_cleanup_date = datetime.now()

    while True:
        area = AREA
        date_today = datetime.now().strftime('%Y/%m-%d')
        current_hour = datetime.now().isoformat()

        # Check if data for the current hour already exists in the database
        query = f"SELECT * FROM prices WHERE time_start <= '{current_hour}' AND time_end > '{current_hour}'"
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

        cpu_cores = psutil.cpu_count(logical=False)
        for i in range(cpu_cores):
            cpu_percent, cpu_freq, cpu_temp, cpu_governor = CPUMonitor.get_cpu_info()
            timestamp = datetime.now().isoformat()
            data_cpu = (timestamp, cpu_cores, i, cpu_percent, cpu_governor, cpu_temp)
            cpu_manager.insert_data(data_cpu)
        print("CPU data inserted successfully.")
        
        # Get the current power cost
        power_cost = price_manager.get_current_price()
        
        # Get the current CPU temperature
        temp = CPUMonitor.get_cpu_temp()
        
        # Get the current CPU usage
        usage = psutil.cpu_percent(interval=1)
        
        # Choose the governor based on the current state
        governor = CPUMonitor.choose_governor(usage, power_cost, temp)
        
        # Set the chosen governor
        set_cpu_governor(governor)
        
        # Check if it's time to remove old data
        if DATA_RETENTION_ENABLED and (datetime.now() - last_cleanup_date).days >= DATA_RETENTION_DAYS:
            # Remove old data
            price_manager.remove_old_data(DATA_RETENTION_DAYS)
            cpu_manager.remove_old_data(DATA_RETENTION_DAYS)

            # Update the last cleanup date
            last_cleanup_date = datetime.now()

        time.sleep(SLEEP_INTERVAL)

if __name__ == "__main__":
    main()
