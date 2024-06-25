import os
import time
import psutil
import cpufreq
from eco2ai import Eco2AI
import nordpool.elspot, nordpool.elbas
import configparser
import sqlite3
import threading
from sklearn.externals import joblib  # Import joblib to load the model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class CPUMonitor:
    def __init__(self, config_file):
        self.config = self.load_config(config_file)
        self.conn = self.setup_database()
        self.eco2ai = Eco2AI(project_name="CPU Monitoring", experiment_description="Monitoring CPU usage and setting governor")
        self.prices_spot = nordpool.elspot.Prices(currency='SEK')

        if os.path.exists('model.pkl'):
            self.model = joblib.load('model.pkl')
        else:
            self.model = self.train_model()

    def load_config(self, config_file):
        config = configparser.ConfigParser()
        config.read(config_file)
        return config

    def setup_database(self):
        conn = sqlite3.connect('cpumonitor.db', check_same_thread=False)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS power_cost (time REAL PRIMARY KEY, cost REAL)''')
        c.execute('''CREATE TABLE IF NOT EXISTS cpu_usage (time REAL PRIMARY KEY, usage REAL)''')
        c.execute('''CREATE TABLE IF NOT EXISTS governor_changes (time REAL, cpu INTEGER, governor TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS energy_consumption (start_time REAL, end_time REAL, consumption REAL, emissions REAL)''')
        c.execute('''CREATE TABLE IF NOT EXISTS available_governors (cpu INTEGER, governor TEXT)''')
        return conn

    def get_cpu_usage(self):
        usage = psutil.cpu_percent(interval=1)
        self.insert_into_db("INSERT INTO cpu_usage VALUES (?, ?)", (time.time(), usage))
        return usage

    def set_governor(self, cpu, usage, power_cost):
        features = [[usage, power_cost]]
        governor = self.predict_governor(features)
        cpufreq.set_governor(cpu, governor)
        self.insert_into_db("INSERT INTO governor_changes VALUES (?, ?, ?)", (time.time(), cpu, governor))

    def get_available_governors(self, cpu):
        governors = self.fetch_data_from_db('SELECT governor FROM available_governors WHERE cpu = ?', (cpu,))
        if governors:
            return [row[0] for row in governors]
        else:
            governors = cpufreq.get_governors(cpu)
            for governor in governors:
                self.insert_into_db("INSERT INTO available_governors VALUES (?, ?)", (cpu, governor))
            return governors

    def log_energy_consumption(self, start_time, end_time):
        self.eco2ai.log(start_time, end_time)
        emissions = self.eco2ai.get_emissions()
        energy_usage = self.eco2ai.get_energy_usage()
        self.insert_into_db("INSERT INTO energy_consumption VALUES (?, ?, ?, ?)", (start_time, end_time, energy_usage, emissions))

    def get_power_cost(self):
        current_time = time.time()
        power_cost = self.fetch_data_from_db('SELECT * FROM power_cost WHERE time = (SELECT MAX(time) FROM power_cost)')
        if power_cost is None or current_time - power_cost[0] >= 3600:
            prices = self.prices_spot.hourly(areas=[self.config['general']['area']])
            current_hour = time.localtime().tm_hour
            power_cost = prices['areas'][self.config['general']['area']]['hours'][current_hour]
            self.insert_into_db("INSERT INTO power_cost VALUES (?, ?)", (current_time, power_cost))
        else:
            power_cost = power_cost[1]

        low_threshold = float(self.config['cost_thresholds']['low'])
        mid_threshold = float(self.config['cost_thresholds']['mid'])
        high_threshold = float(self.config['cost_thresholds']['high'])

        if power_cost <= low_threshold:
            return 'low', power_cost
        elif power_cost <= mid_threshold:
            return 'mid', power_cost
        else:
            return 'high', power_cost

    def monitor_cpu(self, cpu):
        while True:
            start_time = time.time()
            usage = self.get_cpu_usage()
            power_cost_category, power_cost = self.get_power_cost()
            self.set_governor(cpu, usage, power_cost)
            end_time = time.time()
            self.log_energy_consumption(start_time, end_time)
            time.sleep(self.config['general']['sleep_time'])
            print(f"Governor for CPU {cpu} set to {preferred_governor} due to {reason}. Estimated CO2 emissions: {emissions} kg, Estimated energy usage: {energy_usage} kWh.")

    def monitor(self):
        cpus = cpufreq.get_cpus()
        for cpu in cpus:
            threading.Thread(target=self.monitor_cpu, args=(cpu,)).start()

    def insert_into_db(self, sql, params=()):
        with self.conn:
            return self.conn.execute(sql, params)

    def fetch_data_from_db(self, sql, params=()):
        with self.conn:
            return self.conn.execute(sql, params).fetchone()

    def get_moving_average(self, table_name, column_name, window_size=5):
        rows = self.fetch_data_from_db(f'SELECT {column_name} FROM {table_name} ORDER BY time DESC LIMIT ?', (window_size,))
        moving_average = sum(row[0] for row in rows) / len(rows)
        return moving_average

    def predict_governor(self, features):
        return self.model.predict(features)[0]

    def train_model(self):
        rows = self.fetch_data_from_db('SELECT cpu_usage, power_cost, governor FROM training_data')
        X = [[row[0], row[1], self.get_moving_average('cpu_usage', 'usage'), self.get_moving_average('power_cost', 'cost')] for row in rows]
        y = [row[2] for row in rows]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"Model accuracy: {accuracy_score(y_test, y_pred)}")
        joblib.dump(model, 'model.pkl')
        return model

if __name__ == "__main__":
    monitor = CPUMonitor(config_file='cpumonitor.conf')
    monitor.monitor()
