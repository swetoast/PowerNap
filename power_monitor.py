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
from sklearn.datasets import make_classification

class CPUMonitor:
    def __init__(self, config_file):
        # Load the configuration file and setup the database
        self.config = self.load_config(config_file)
        self.conn = self.setup_database()

        # Initialize Eco2AI and Nordpool for energy monitoring and cost calculation
        self.eco2ai = Eco2AI(project_name="CPU Monitoring", experiment_description="Monitoring CPU usage and setting governor")
        self.prices_spot = nordpool.elspot.Prices(currency='SEK')

        # Load the trained model or train a new one if it does not exist
        if os.path.exists('model.pkl'):
            self.model = joblib.load('model.pkl')  # Load the trained model
        else:
            self.model = self.train_model()  # Train a new model

    def load_config(self, config_file):
        # Load the configuration file
        config = configparser.ConfigParser()
        config.read(config_file)
        return config

    def setup_database(self):
        # Setup the SQLite database
        conn = sqlite3.connect('cpumonitor.db', check_same_thread=False)
        c = conn.cursor()

        # Create the necessary tables if they do not exist
        c.execute('''CREATE TABLE IF NOT EXISTS power_cost (time REAL PRIMARY KEY, cost REAL)''')
        c.execute('''CREATE TABLE IF NOT EXISTS cpu_usage (time REAL PRIMARY KEY, usage REAL)''')
        c.execute('''CREATE TABLE IF NOT EXISTS governor_changes (time REAL, cpu INTEGER, governor TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS energy_consumption (start_time REAL, end_time REAL, consumption REAL, emissions REAL)''')
        c.execute('''CREATE TABLE IF NOT EXISTS available_governors (cpu INTEGER, governor TEXT)''')  # New table for available governors

        return conn

    def get_cpu_usage(self):
        # Get the current CPU usage and log it in the database
        usage = psutil.cpu_percent(interval=1)
        self.execute("INSERT INTO cpu_usage VALUES (?, ?)", (time.time(), usage))
        return usage

    def set_governor(self, cpu, usage, power_cost):
        # Set the CPU governor based on the prediction of the machine learning model
        # Prepare the features for the model
        features = [[usage, power_cost]]

        # Predict the governor using the model
        governor = self.model.predict(features)[0]

        # Set the governor
        cpufreq.set_governor(cpu, governor)
        self.execute("INSERT INTO governor_changes VALUES (?, ?, ?)", (time.time(), cpu, governor))

    def get_available_governors(self, cpu):
        # Get the available governors for a given CPU from the database
        # Fetch the available governors from the database
        self.c.execute('SELECT governor FROM available_governors WHERE cpu = ?', (cpu,))
        rows = self.c.fetchall()

        if rows:
            # If the governors are in the database, return them
            return [row[0] for row in rows]
        else:
            # If the governors are not in the database, fetch them using cpufreq, store them in the database, and return them
            governors = cpufreq.get_governors(cpu)
            for governor in governors:
                self.execute("INSERT INTO available_governors VALUES (?, ?)", (cpu, governor))
            return governors

    def log_energy_consumption(self, start_time, end_time):
        # Log the energy consumption for a given time interval
        # Log the energy consumption using Eco2AI
        self.eco2ai.log(start_time, end_time)

        # Get the estimated emissions and energy usage
        emissions = self.eco2ai.get_emissions()
        energy_usage = self.eco2ai.get_energy_usage()

        # Insert the energy consumption and emissions into the database
        self.execute("INSERT INTO energy_consumption VALUES (?, ?, ?, ?)", (start_time, end_time, energy_usage, emissions))

    def get_power_cost(self):
        # Get the current power cost
        current_time = time.time()

        # Fetch the latest power cost from the database
        self.c.execute('SELECT * FROM power_cost WHERE time = (SELECT MAX(time) FROM power_cost)')
        row = self.fetchone()

        if row is None or current_time - row[0] >= 3600:
            # If an hour has passed since the last fetch, fetch the power cost from Nordpool
            prices = self.prices_spot.hourly(areas=[self.config['general']['area']])
            current_hour = time.localtime().tm_hour
            power_cost = prices['areas'][self.config['general']['area']]['hours'][current_hour]

            # Insert the new power cost into the database
            self.execute("INSERT INTO power_cost VALUES (?, ?)", (current_time, power_cost))
        else:
            power_cost = row[1]

        # Categorize the power cost into low, mid, or high
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
        # Monitor the CPU usage and adjust the governor for a single CPU
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
        # Create a separate thread for each CPU
        cpus = cpufreq.get_cpus()
        for cpu in cpus:
            threading.Thread(target=self.monitor_cpu, args=(cpu,)).start()

    def execute(self, sql, params=()):
        # Execute a SQL statement
        with self.conn:
            return self.conn.execute(sql, params)

    def train_model(self):
        # Fetch the data from the SQLite database
        self.c.execute('SELECT cpu_usage, power_cost, governor FROM training_data')
        rows = self.c.fetchall()

        # Split the data into features (X) and target variable (y)
        X = [[row[0], row[1]] for row in rows]
        y = [row[2] for row in rows]

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a Random Forest classifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test)
        print(f"Model accuracy: {accuracy_score(y_test, y_pred)}")

        # Save the trained model
        joblib.dump(model, 'model.pkl')

        return model

if __name__ == "__main__":
    # Create an instance of CPUMonitor and start monitoring
    monitor = CPUMonitor(config_file='cpumonitor.conf')
    monitor.monitor()
