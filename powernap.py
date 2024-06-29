import os
import configparser
import logging
import logging.handlers
import sqlite3
import threading
import time
from queue import Queue
import requests
import datetime
import psutil
from joblib import load, dump
from sklearn import ensemble, metrics, model_selection

class GovernorManager:
    @staticmethod
    def set_cpu_governor(cpu, governor):
        with open(f'/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor', 'w') as f:
            f.write(governor)

    @staticmethod
    def get_available_governors(cpu):
        with open(f'/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_available_governors', 'r') as f:
            return [g for g in f.read().strip().split(' ') if g not in ['userspace', 'schedutil']]

    @staticmethod
    def get_current_governor(cpu):
        with open(f'/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor', 'r') as f:
            return f.read().strip()

class ConfigLoader:
    @staticmethod
    def load_config(config_file):
        config = configparser.ConfigParser()
        config.read(os.path.join(os.path.dirname(os.path.realpath(__file__)), config_file))
        return config

class DatabaseManager:
    def __init__(self, db_name):
        self.conn = sqlite3.connect(db_name, check_same_thread=False)
        self.setup_database()
        self.queue = Queue()
        threading.Thread(target=self._process_queue).start()

    def setup_database(self):
        c = self.conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS cpu_usage (time REAL PRIMARY KEY, usage REAL)''')
        c.execute('''CREATE TABLE IF NOT EXISTS power_cost (time REAL PRIMARY KEY, cost REAL)''')
        c.execute('''CREATE TABLE IF NOT EXISTS governor_changes (time REAL, cpu INTEGER, governor TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS available_governors (cpu INTEGER, governor TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS training_data (cpu_usage REAL, governor TEXT)''') 

    def _process_queue(self):
        while True:
            sql, params = self.queue.get()
            self.conn.execute(sql, params)
            self.conn.commit()
            self.queue.task_done()

    def insert_into_db(self, sql, params=()):
        self.queue.put((sql, params))

    def fetch_data_from_db(self, sql, params=()):
        return self.conn.execute(sql, params).fetchone()

    def purge_old_data(self, days):
        cutoff_time = time.time() - days * 24 * 60 * 60
        self.conn.execute("DELETE FROM cpu_usage WHERE time < ?", (cutoff_time,))
        self.conn.execute("DELETE FROM governor_changes WHERE time < ?", (cutoff_time,))
        self.conn.execute("DELETE FROM power_cost WHERE time < ?", (cutoff_time,))
        self.conn.commit()

class ModelManager:
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.model = load('model.pkl') if os.path.exists('model.pkl') else self.train_model()

    def train_model(self):
        rows = self.db_manager.fetch_data_from_db('SELECT cpu_usage, power_cost, governor FROM training_data')
        if rows is None:
            print("No training data available.")
            return
        X = [[row[0], row[1]] for row in rows]  # Include power_cost in the feature set
        y = [row[2] for row in rows]

        # Compute the min and max for each feature
        min_cpu_usage = min(x[0] for x in X)
        max_cpu_usage = max(x[0] for x in X)
        min_power_cost = min(x[1] for x in X)
        max_power_cost = max(x[1] for x in X)

        # Normalize the features
        X = [[(x[0] - min_cpu_usage) / (max_cpu_usage - min_cpu_usage), (x[1] - min_power_cost) / (max_power_cost - min_power_cost)] for x in X]

        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)
        model = ensemble.RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"Model accuracy: {metrics.accuracy_score(y_test, y_pred)}")
        dump(model, 'model.pkl')
        return model

    def predict_governor(self, features):
        return 'performance' if self.model is None else self.model.predict(features)[0]

class CPUMonitor:
    def __init__(self, config_file):
        self.config = ConfigLoader.load_config(config_file)
        self.db_manager = DatabaseManager('monitor.db')

        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.handlers.SysLogHandler(address = '/dev/log')
        self.logger.addHandler(handler)

    def get_cpus(self):
        return list(range(os.cpu_count()))

    def get_cpu_usage(self):
        usage = psutil.cpu_percent(interval=1)
        self.db_manager.insert_into_db("INSERT INTO cpu_usage VALUES (?, ?)", (time.time(), usage))
        return usage

    def get_power_cost(self):
        current_time = datetime.datetime.now()
        formatted_time = current_time.strftime('%Y/%m-%d')  # Format the date as YYYY/MM-DD
        response = requests.get(f'https://www.elprisetjustnu.se/api/v1/prices/{formatted_time}_SE3.json')
        
        if response.status_code == 200 and response.text.strip():  # Check if the response is OK and not empty
            try:
                data = response.json()
            except json.JSONDecodeError:
                print("Error decoding JSON response")
                return None
        else:
            print("Error fetching power cost data")
            return None

        # Find the current hour's data
        for hour_data in data:
            time_start = hour_data['time_start']
            time_end = hour_data['time_end']
            if time_start <= current_time.isoformat() < time_end:
                power_cost = hour_data['SEK_per_kWh']
                self.db_manager.insert_into_db("INSERT INTO power_cost VALUES (?, ?)", (time.time(), power_cost))
                
                thresholds = {
                    'low': float(self.config['cost_thresholds']['low']),
                    'mid': float(self.config['cost_thresholds']['mid']),
                    'high': float(self.config['cost_thresholds']['high'])
                }

                for category, threshold in thresholds.items():
                    if power_cost is not None and power_cost <= threshold:
                        return category, power_cost

                return 'high', power_cost

        return None  # Return None if no matching hour is found

    def set_governor(self, cpu, usage, power_cost):
        governors = GovernorManager.get_available_governors(cpu)
        if governors:
            if power_cost is not None:
                if usage > 75 and power_cost < 1/3:
                    governor = 'performance'
                elif usage < 25 and power_cost > 2/3:
                    governor = 'powersave'
                elif 25 <= usage <= 75 and 1/3 <= power_cost <= 2/3:
                    governor = 'conservative'
                else:
                    governor = 'ondemand'
            else:
                governor = 'ondemand'  # default governor when power cost is None
            
            if governor in governors:
                GovernorManager.set_cpu_governor(cpu, governor)
                self.db_manager.insert_into_db("INSERT INTO governor_changes VALUES (?, ?, ?)", (time.time(), cpu, governor))

    def monitor_cpu(self, cpu):
        while True:
            start_time = time.time()
            usage = self.get_cpu_usage()
            power_cost_category, power_cost = self.get_power_cost()
            self.set_governor(cpu, usage, power_cost)
            time.sleep(int(self.config['general']['sleep_time']))  # Convert sleep_time to int

    def monitor(self):
        cpus = self.get_cpus()
        print(f"Number of CPUs: {len(cpus)}")
        for cpu in cpus:
            print(f"CPU{cpu} initial governor: {GovernorManager.get_current_governor(cpu)}")
        initial_cpu_usage = self.get_cpu_usage()
        print(f"Initial CPU usage: {initial_cpu_usage}%")
        initial_power_cost_category, initial_power_cost = self.get_power_cost()
        print(f"Initial power cost: {initial_power_cost} ({initial_power_cost_category})")

        purge_enabled = self.config['database']['purge_enabled'].lower() == 'yes'
        if purge_enabled:
            purge_after_days = int(self.config['database']['purge_after_days'])
            self.db_manager.purge_old_data(purge_after_days)
        for cpu in cpus:
            threading.Thread(target=self.monitor_cpu, args=(cpu,)).start()

if __name__ == "__main__":
    monitor = CPUMonitor(config_file='powernap.conf')
    monitor.monitor()
