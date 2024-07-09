import os
import configparser
import sqlite3
import threading
import time
from queue import Queue
import requests
import datetime
import psutil
from joblib import load, dump
from sklearn import ensemble, metrics, model_selection
from imblearn.over_sampling import SMOTE

class Utils:
    @staticmethod
    def get_current_time():
        return time.time()

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
        try:
            config.read(os.path.join(os.path.dirname(os.path.realpath(__file__)), config_file))
        except configparser.Error as e:
            print(f"Failed to load config file: {e}")
            raise
        return config

class DatabaseManager:
    def __init__(self, db_name):
        # Create the full path to the monitor.db file
        db_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), db_name)

        try:
            self.conn = sqlite3.connect(db_path, check_same_thread=False)
            self.setup_database()
            self.queue = Queue()
            threading.Thread(target=self._process_queue).start()
        except sqlite3.Error as e:
            print(f"Failed to connect to the database: {e}")
            raise

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
            try:
                self.conn.execute(sql, params)
                self.conn.commit()
            except sqlite3.Error as e:
                print(f"Failed to execute SQL command: {e}")
            finally:
                self.queue.task_done()

    def insert_into_db(self, sql, params=()):
        self.queue.put((sql, params))

    def fetch_data_from_db(self, sql, params=()):
        try:
            return self.conn.execute(sql, params).fetchone()
        except sqlite3.Error as e:
            print(f"Failed to fetch data from database: {e}")
            return None

    def purge_old_data(self, days):
        cutoff_time = Utils.get_current_time() - days * 24 * 60 * 60
        try:
            self.conn.execute("DELETE FROM cpu_usage WHERE time < ?", (cutoff_time,))
            self.conn.execute("DELETE FROM governor_changes WHERE time < ?", (cutoff_time,))
            self.conn.execute("DELETE FROM power_cost WHERE time < ?", (cutoff_time,))
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"Failed to purge old data: {e}")

class ModelManager:
    def __init__(self, db_manager, model_file):
        self.db_manager = db_manager

        # Create the full path to the model.pkl file
        model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), model_file)

        try:
            self.model = load(model_path) if os.path.exists(model_path) else self.train_model()
        except Exception as e:
            print(f"Failed to load or train the model: {e}")
            self.model = None

    def train_model(self):
        rows = self.db_manager.fetch_data_from_db('SELECT cpu_usage, power_cost, governor FROM training_data WHERE governor NOT IN (?, ?)', ('userspace', 'schedutil'))
        if rows is None:
            print("No training data available.")
            return
        print(f"Number of training examples: {len(rows)}")
        X = [[row[0], row[1]] for row in rows]  # Include power_cost in the feature set
        y = [row[2] for row in rows]

        # Compute the min and max for each feature
        min_cpu_usage = min(x[0] for x in X)
        max_cpu_usage = max(x[0] for x in X)
        min_power_cost = min(x[1] for x in X)
        max_power_cost = max(x[1] for x in X)

        # Normalize the features
        X = [[(x[0] - min_cpu_usage) / (max_cpu_usage - min_cpu_usage), (x[1] - min_power_cost) / (max_power_cost - min_power_cost)] for x in X]

        try:
            X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)
            sm = SMOTE(random_state=42)
            X_res, y_res = sm.fit_resample(X_train, y_train)
            model = ensemble.RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_res, y_res)
            y_pred = model.predict(X_test)
            print(f"Model accuracy: {metrics.accuracy_score(y_test, y_pred)}")
            dump(model, 'model.pkl')
            return model
        except Exception as e:
            print(f"Failed to train the model: {e}")
            return None

    def predict_governor(self, features):
        if self.model is None:
            print("Model is not available. Returning default governor 'performance'.")
            return 'performance'
        else:
            return self.model.predict(features)[0]

class CPUMonitor:
    def __init__(self, config_file):
        try:
            self.config = ConfigLoader.load_config(config_file)
        except configparser.Error as e:
            print(f"Failed to load config file: {e}")
            raise
        self.db_manager = DatabaseManager('monitor.db')
        self.power_cost_data = None

    def get_cpus(self):
        return list(range(os.cpu_count()))

    def get_cpu_usage(self):
        usage = psutil.cpu_percent(interval=1)
        self.db_manager.insert_into_db("INSERT INTO cpu_usage VALUES (?, ?)", (Utils.get_current_time(), usage))
        return usage

    def get_power_cost(self):
        current_time = datetime.datetime.now()
        formatted_time = current_time.strftime('%Y/%m-%d')  # Format the date as YYYY/MM-DD
        area = self.config['general']['area']  # Read the area from the config file

        # Check if the data is already fetched
        if self.power_cost_data is None or not self.is_data_available(current_time):
            retry_count = 0
            while True:
                try:
                    response = requests.get(f'https://www.elprisetjustnu.se/api/v1/prices/{formatted_time}_{area}.json', timeout=10)
                    response.raise_for_status()  # Raises a HTTPError if the response status is 4xx, 5xx
                    self.power_cost_data = response.json()
                    break
                except requests.exceptions.RequestException as e:
                    retry_count += 1
                    if retry_count <= 3:  # Only print the error message for the first 3 retries
                        print("Site unavailable at the moment, retrying in 20 minutes...")
                    time.sleep(1200)  # Wait for 5 minutes before retrying

        # Find the current hour's data
        for hour_data in self.power_cost_data:
            time_start = hour_data['time_start']
            time_end = hour_data['time_end']
            if time_start <= current_time.isoformat() < time_end:
                power_cost = hour_data['SEK_per_kWh']
                self.db_manager.insert_into_db("INSERT INTO power_cost VALUES (?, ?)", (Utils.get_current_time(), power_cost))
                
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

    def is_data_available(self, current_time):
        for hour_data in self.power_cost_data:
            time_start = hour_data['time_start']
            time_end = hour_data['time_end']
            if time_start <= current_time.isoformat() < time_end:
                return True
        return False

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

            # Exclude 'userspace' and 'schedutil'
            if governor in ['userspace', 'schedutil']:
                print(f"Governor {governor} is not allowed. Setting to 'ondemand'.")
                governor = 'ondemand'

            current_governor = GovernorManager.get_current_governor(cpu)
            if governor != current_governor and governor in governors:
                GovernorManager.set_cpu_governor(cpu, governor)
                self.db_manager.insert_into_db("INSERT INTO governor_changes VALUES (?, ?, ?)", (Utils.get_current_time(), cpu, governor))
                print(f"Governor for CPU{cpu} set to {governor}")

    def monitor_cpu(self, cpu):
        while True:
            start_time = Utils.get_current_time()
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
