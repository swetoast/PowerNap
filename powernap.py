#!/usr/bin/env python3
import os
import configparser
import logging
import logging.handlers
import sqlite3
import threading
import time
from queue import Queue

import nordpool.elbas
import nordpool.elspot
import psutil
from joblib import load, dump
from sklearn import ensemble, metrics, model_selection

def normalize_feature(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value)

class GovernorManager:
    @staticmethod
    def set_cpu_governor(cpu, governor):
        try:
            with open(f'/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor', 'w') as f:
                f.write(governor)
        except IOError as e:
            print(f"Error setting governor: {e}")

    @staticmethod
    def get_available_governors(cpu):
        try:
            with open(f'/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_available_governors', 'r') as f:
                governors = f.read().strip().split(' ')
                governors = [g for g in governors if g not in ['userspace', 'schedutil']]
                return governors
        except IOError as e:
            print(f"Error getting available governors: {e}")

class ConfigLoader:
    @staticmethod
    def load_config(config_file):
        config = configparser.ConfigParser()
        config.read(config_file)
        return config

class DatabaseManager:
    def __init__(self, db_name):
        try:
            self.conn = sqlite3.connect(db_name, check_same_thread=False)
            self.setup_database()
            self.queue = Queue()
            threading.Thread(target=self._process_queue).start()
        except Exception as e:
            print(f"Failed to connect to the database: {e}")
            raise

    def setup_database(self):
        c = self.conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS power_cost (time REAL PRIMARY KEY, cost REAL)''')
        c.execute('''CREATE TABLE IF NOT EXISTS cpu_usage (time REAL PRIMARY KEY, usage REAL)''')
        c.execute('''CREATE TABLE IF NOT EXISTS governor_changes (time REAL, cpu INTEGER, governor TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS available_governors (cpu INTEGER, governor TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS training_data (cpu_usage REAL, power_cost REAL, governor TEXT)''') 

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
        self.conn.execute("DELETE FROM power_cost WHERE time < ?", (cutoff_time,))
        self.conn.execute("DELETE FROM cpu_usage WHERE time < ?", (cutoff_time,))
        self.conn.execute("DELETE FROM governor_changes WHERE time < ?", (cutoff_time,))
        self.conn.commit()

class ModelManager:
    def __init__(self, db_manager):
        self.db_manager = db_manager
        try:
            if os.path.exists('model.pkl'):
                self.model = load('model.pkl')
            else:
                self.model = self.train_model()
        except Exception as e:
            print(f"Failed to train or load the model: {e}")
            raise

    def train_model(self):
        rows = self.db_manager.fetch_data_from_db('SELECT cpu_usage, power_cost, governor FROM training_data')
        if rows is None:
            print("No training data available.")
            return
        X = [[row[0], row[1]] for row in rows]
        y = [row[2] for row in rows]

        # Compute the min and max for each feature
        min_cpu_usage = min(x[0] for x in X)
        max_cpu_usage = max(x[0] for x in X)
        min_power_cost = min(x[1] for x in X)
        max_power_cost = max(x[1] for x in X)

        # Normalize the features
        X = [[normalize_feature(x[0], min_cpu_usage, max_cpu_usage), 
              normalize_feature(x[1], min_power_cost, max_power_cost)] for x in X]

        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)
        model = ensemble.RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"Model accuracy: {metrics.accuracy_score(y_test, y_pred)}")
        dump(model, 'model.pkl')
        return model

    def predict_governor(self, features):
        if self.model is None:
            # Return 'performance' as the default governor
            return 'performance'
        else:
            # Normalize the features
            features = [[normalize_feature(features[0][0], min_cpu_usage, max_cpu_usage), 
                         normalize_feature(features[0][1], min_power_cost, max_power_cost)]]
            return self.model.predict(features)[0]

class CPUMonitor:
    def __init__(self, config_file):
        self.config = ConfigLoader.load_config(config_file)
        self.currency = self.config['currency']['value']  # Load the currency from the configuration
        self.db_manager = DatabaseManager('monitor.db')
        self.model_manager = ModelManager(self.db_manager)
        self.prices_spot = nordpool.elspot.Prices(currency=self.currency)  # Use the currency from the configuration

        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.handlers.SysLogHandler(address = '/dev/log')
        self.logger.addHandler(handler)

    def get_cpus(self):
        # Get the number of available CPUs
        num_cpus = os.cpu_count()

        # Return a list of CPUs
        return list(range(num_cpus))

    def get_cpu_usage(self):
        usage = psutil.cpu_percent(interval=1)
        self.db_manager.insert_into_db("INSERT INTO cpu_usage VALUES (?, ?)", (time.time(), usage))
        return usage

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

    def get_power_cost(self):
        current_time = time.time()
        power_cost = self.db_manager.fetch_data_from_db('SELECT * FROM power_cost WHERE time = (SELECT MAX(time) FROM power_cost)')
        if power_cost is None or len(power_cost) == 0 or current_time - power_cost[0] >= 3600:
            prices = self.prices_spot.hourly(areas=[self.config['general']['area']])
            current_hour = time.localtime().tm_hour
            if 'hours' in prices['areas'][self.config['general']['area']]:
                power_cost = prices['areas'][self.config['general']['area']]['hours'][current_hour]
                self.db_manager.insert_into_db("INSERT INTO power_cost VALUES (?, ?)", (current_time, power_cost))
            else:
                power_cost = None  # You can set a default value here
        else:
            power_cost = power_cost[1]

        thresholds = {
            'low': float(self.config['cost_thresholds']['low']),
            'mid': float(self.config['cost_thresholds']['mid']),
            'high': float(self.config['cost_thresholds']['high'])
        }

        for category, threshold in thresholds.items():
            if power_cost is not None and power_cost <= threshold:
                return category, power_cost

        return 'high', power_cost

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
            print(f"CPU{cpu} initial governor: {GovernorManager.get_available_governors(cpu)}")
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
