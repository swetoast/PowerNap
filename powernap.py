#!/usr/bin/env python3
"""
PowerNap monitors Linux CPU load, electricity prices, and thermal conditions to
choose an appropriate CPU frequency governor.

The script runs as a long-lived daemon loop. Each cycle fetches or reuses cached
electricity price data, samples per-core CPU usage, smooths recent load, classifies
the current workload, evaluates the current price against the day's price range,
checks CPU temperature, and calculates a governor decision.

Decisions are made through a layered model:

- safety overrides for critical thermal conditions
- workload classification for idle, light, balanced, burst, sustained, and I/O-bound load
- electricity price classification using daily price rank and lookahead
- a policy matrix that provides the baseline governor
- a formulaic aggression score that can adjust the baseline
- transition gating to avoid rapid governor changes

Runtime history is stored in SQLite, including electricity prices, CPU samples,
and governor change events. PrettyTable is used only by the explicit report
command to present stored history in readable tables.

PowerNap does not expose a web server or remote control interface. It is intended
to run locally through the CLI or as a systemd service.
"""
from __future__ import annotations

import argparse
import configparser
import glob
import json
import logging
import os
import random
import signal
import sqlite3
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from statistics import mean, median
from typing import Iterable, Optional

import psutil
import requests
from prettytable import PrettyTable

APP_NAME = "PowerNap"
DEFAULT_CONFIG_NAME = "powernap.conf"
STOP_REQUESTED = False

GOVERNOR_PRIORITY = {"powersave": 0, "conservative": 1, "schedutil": 2, "performance": 3}
PRIORITY_GOVERNOR = {v: k for k, v in GOVERNOR_PRIORITY.items()}
PREFERRED_GOVERNORS = ("powersave", "conservative", "schedutil", "performance")


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def request_stop(signum, _frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    logging.info("Received signal %s; stopping after current cycle.", signum)


class WorkloadClass(str, Enum):
    IDLE = "idle"
    LIGHT = "light"
    BALANCED = "balanced"
    BURST = "burst"
    SUSTAINED_HEAVY = "sustained_heavy"
    IO_BOUND = "io_bound"


class PriceClass(str, Enum):
    CHEAP = "cheap"
    NORMAL = "normal"
    EXPENSIVE = "expensive"
    UNKNOWN = "unknown"


class ThermalClass(str, Enum):
    NORMAL = "normal"
    WARM = "warm"
    HOT = "hot"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Config:
    area: str = "SE3"
    sleep_interval: int = 5
    database_dir: str = "."
    log_level: str = "INFO"
    data_retention_enabled: bool = True
    data_retention_days: int = 30
    commit_interval_minutes: int = 5
    usage_method: str = "median"
    request_timeout_sec: int = 10
    smooth_factor: float = 0.30
    history_samples: int = 24
    high_load_threshold: float = 70.0
    burst_cpu_threshold: float = 75.0
    idle_cpu_threshold: float = 15.0
    iowait_threshold: float = 25.0
    cheap_rank_max: float = 0.25
    expensive_rank_min: float = 0.80
    lookahead_hours: int = 3
    lookahead_delta: float = 0.20
    cpu_weight: float = 0.55
    urgency_weight: float = 0.20
    cheap_now_weight: float = 0.15
    lookahead_weight: float = 0.10
    thermal_weight: float = 0.25
    thermal_enabled: bool = True
    warm_temp: float = 70.0
    hot_temp: float = 80.0
    critical_temp: float = 90.0
    min_hold_sec: int = 60
    upscale_cooldown_sec: int = 15
    score_margin: float = 7.5
    dry_run: bool = False


def _strip(value: str) -> str:
    return value.split("#", 1)[0].strip()


def load_config(path: Path, dry_run_override: bool = False) -> Config:
    parser = configparser.ConfigParser()
    parser.optionxform = str
    parser.read(path)

    def get(section: str, option: str, default, cast):
        try:
            raw = parser.get(section, option)
        except (configparser.NoSectionError, configparser.NoOptionError):
            return default
        if cast is bool:
            return parser.getboolean(section, option)
        if cast is str:
            return _strip(raw)
        return cast(_strip(raw))

    cfg = Config(
        area=get("AreaCode", "AREA", Config.area, str),
        sleep_interval=max(1, get("Runtime", "SLEEP_INTERVAL", Config.sleep_interval, int)),
        database_dir=get("Storage", "DATABASE_DIR", Config.database_dir, str),
        log_level=get("Logging", "LEVEL", Config.log_level, str).upper(),
        data_retention_enabled=get("DataRetention", "ENABLED", Config.data_retention_enabled, bool),
        data_retention_days=max(1, get("DataRetention", "DAYS", Config.data_retention_days, int)),
        commit_interval_minutes=max(0, get("Runtime", "COMMIT_INTERVAL_MINUTES", Config.commit_interval_minutes, int)),
        usage_method=get("Runtime", "USAGE_METHOD", Config.usage_method, str).lower(),
        request_timeout_sec=max(2, get("Network", "REQUEST_TIMEOUT_SEC", Config.request_timeout_sec, int)),
        smooth_factor=get("Metrics", "SMOOTH_FACTOR", Config.smooth_factor, float),
        history_samples=max(6, get("Metrics", "HISTORY_SAMPLES", Config.history_samples, int)),
        high_load_threshold=get("Metrics", "HIGH_LOAD_THRESHOLD", Config.high_load_threshold, float),
        burst_cpu_threshold=get("Metrics", "BURST_CPU_THRESHOLD", Config.burst_cpu_threshold, float),
        idle_cpu_threshold=get("Metrics", "IDLE_CPU_THRESHOLD", Config.idle_cpu_threshold, float),
        iowait_threshold=get("Metrics", "IOWAIT_THRESHOLD", Config.iowait_threshold, float),
        cheap_rank_max=clamp(get("PriceStrategy", "CHEAP_RANK_MAX", Config.cheap_rank_max, float)),
        expensive_rank_min=clamp(get("PriceStrategy", "EXPENSIVE_RANK_MIN", Config.expensive_rank_min, float)),
        lookahead_hours=max(1, get("PriceStrategy", "LOOKAHEAD_HOURS", Config.lookahead_hours, int)),
        lookahead_delta=clamp(get("PriceStrategy", "LOOKAHEAD_DELTA", Config.lookahead_delta, float)),
        cpu_weight=get("Formula", "CPU_WEIGHT", Config.cpu_weight, float),
        urgency_weight=get("Formula", "URGENCY_WEIGHT", Config.urgency_weight, float),
        cheap_now_weight=get("Formula", "CHEAP_NOW_WEIGHT", Config.cheap_now_weight, float),
        lookahead_weight=get("Formula", "LOOKAHEAD_WEIGHT", Config.lookahead_weight, float),
        thermal_weight=get("Formula", "THERMAL_WEIGHT", Config.thermal_weight, float),
        thermal_enabled=get("Thermal", "ENABLED", Config.thermal_enabled, bool),
        warm_temp=get("Thermal", "WARM_TEMP", Config.warm_temp, float),
        hot_temp=get("Thermal", "HOT_TEMP", Config.hot_temp, float),
        critical_temp=get("Thermal", "CRITICAL_TEMP", Config.critical_temp, float),
        min_hold_sec=max(0, get("Transition", "MIN_HOLD_SEC", Config.min_hold_sec, int)),
        upscale_cooldown_sec=max(0, get("Transition", "UPSCALE_COOLDOWN_SEC", Config.upscale_cooldown_sec, int)),
        score_margin=max(0.0, get("Transition", "SCORE_MARGIN", Config.score_margin, float)),
        dry_run=dry_run_override or get("Runtime", "DRY_RUN", Config.dry_run, bool),
    )
    if cfg.usage_method not in ("median", "average", "max"):
        cfg = Config(**{**cfg.__dict__, "usage_method": "median"})
    if not (0 < cfg.smooth_factor <= 1):
        cfg = Config(**{**cfg.__dict__, "smooth_factor": 0.30})
    return cfg


@dataclass
class Metrics:
    timestamp: str
    cpu_usage: float
    smoothed_cpu: float
    loadavg_1m_ratio: float
    iowait_percent: float
    sustained_load_ratio: float
    cpu_temp: Optional[float]
    current_price: Optional[float]
    price_rank: Optional[float]
    future_price_rank: Optional[float]
    price_trend: Optional[float]


@dataclass
class Decision:
    governor: str
    state: str
    score: float
    workload_class: str
    price_class: str
    thermal_class: str
    load_score: float
    urgency_score: float
    cheap_now_score: float
    lookahead_score: float
    thermal_penalty: float
    baseline_governor: str
    reason: str
    safety_override: bool = False


class DatabaseManager:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA temp_store=MEMORY")

    def execute(self, query: str, params: Iterable = (), commit: bool = False):
        try:
            cur = self.conn.execute(query, tuple(params))
            if commit:
                self.conn.commit()
            return cur.fetchall()
        except sqlite3.Error as exc:
            logging.error("SQLite error in %s: %s", self.path, exc)
            logging.debug("Query: %s", query)
            return []

    def executemany(self, query: str, rows: Iterable[Iterable], commit: bool = False):
        try:
            cur = self.conn.executemany(query, rows)
            if commit:
                self.conn.commit()
            return cur.fetchall()
        except sqlite3.Error as exc:
            logging.error("SQLite batch error in %s: %s", self.path, exc)
            logging.debug("Query: %s", query)
            return []

    def close(self) -> None:
        self.conn.close()


def parse_price_time(value: str) -> int:
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp())


class PriceManager(DatabaseManager):
    def __init__(self, path: Path):
        super().__init__(path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS prices (
                id INTEGER PRIMARY KEY,
                SEK_per_kWh REAL NOT NULL,
                time_start TEXT NOT NULL,
                time_end TEXT NOT NULL,
                start_ts INTEGER NOT NULL,
                end_ts INTEGER NOT NULL,
                UNIQUE(time_start, time_end)
            )
        """)
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_prices_epoch ON prices(start_ts, end_ts)")
        self.conn.commit()

    def insert_bulk(self, items: list[dict]) -> None:
        rows = []
        for item in items:
            try:
                rows.append((float(item["SEK_per_kWh"]), item["time_start"], item["time_end"], parse_price_time(item["time_start"]), parse_price_time(item["time_end"])))
            except Exception as exc:
                logging.warning("Skipping bad price row %r: %s", item, exc)
        self.executemany("""
            INSERT OR IGNORE INTO prices(SEK_per_kWh, time_start, time_end, start_ts, end_ts)
            VALUES (?, ?, ?, ?, ?)
        """, rows, commit=True)

    def current_price(self, now_ts: Optional[int] = None) -> Optional[float]:
        now_ts = int(time.time()) if now_ts is None else now_ts
        rows = self.execute("""
            SELECT SEK_per_kWh FROM prices
            WHERE start_ts <= ? AND end_ts > ?
            ORDER BY start_ts DESC LIMIT 1
        """, (now_ts, now_ts))
        return float(rows[0]["SEK_per_kWh"]) if rows else None

    def has_date(self, d: date) -> bool:
        rows = self.execute("SELECT 1 FROM prices WHERE time_start LIKE ? LIMIT 1", (d.isoformat() + "%",))
        return bool(rows)

    def day_prices(self, d: date) -> list[sqlite3.Row]:
        local_tz = datetime.now().astimezone().tzinfo
        start = int(datetime.combine(d, datetime.min.time()).replace(tzinfo=local_tz).timestamp())
        end = start + 86400
        return self.execute("SELECT * FROM prices WHERE start_ts >= ? AND start_ts < ? ORDER BY start_ts", (start, end))

    def price_rank(self, price: Optional[float], d: date) -> Optional[float]:
        if price is None:
            return None
        values = [float(r["SEK_per_kWh"]) for r in self.day_prices(d)]
        if not values:
            return None
        low, high = min(values), max(values)
        if high <= low:
            return 0.5
        return clamp((price - low) / (high - low))

    def future_average_rank(self, d: date, hours: int) -> Optional[float]:
        now_ts = int(time.time())
        end_ts = now_ts + hours * 3600
        future = self.execute("""
            SELECT SEK_per_kWh FROM prices
            WHERE start_ts >= ? AND start_ts < ?
            ORDER BY start_ts
        """, (now_ts, end_ts))
        if not future:
            return None
        avg = mean(float(r["SEK_per_kWh"]) for r in future)
        return self.price_rank(avg, d)

    def prune(self, days: int) -> None:
        cutoff = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp())
        self.execute("DELETE FROM prices WHERE end_ts < ?", (cutoff,), commit=True)


class CPUHistory(DatabaseManager):
    def __init__(self, path: Path, core_count: int):
        super().__init__(path)
        self.core_count = core_count
        self.buffer = {i: deque(maxlen=900) for i in range(core_count)}
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS cpu_usage (
                id INTEGER PRIMARY KEY,
                timestamp TEXT NOT NULL,
                timestamp_ts INTEGER NOT NULL,
                cpu_cores INTEGER NOT NULL,
                cpu_core_id INTEGER NOT NULL,
                cpu_usage REAL NOT NULL,
                cpu_governor TEXT NOT NULL
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS governor_events (
                id INTEGER PRIMARY KEY,
                timestamp TEXT NOT NULL,
                timestamp_ts INTEGER NOT NULL,
                old_governor TEXT NOT NULL,
                new_governor TEXT NOT NULL,
                state TEXT NOT NULL,
                score REAL NOT NULL,
                reason TEXT NOT NULL,
                decision_json TEXT NOT NULL
            )
        """)
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_cpu_usage_ts ON cpu_usage(timestamp_ts)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_governor_events_ts ON governor_events(timestamp_ts)")
        self.conn.commit()

    def add_sample(self, core_id: int, usage: float) -> None:
        if core_id in self.buffer:
            self.buffer[core_id].append(float(usage))

    def commit_usage(self, governor: str) -> None:
        now = datetime.now().astimezone()
        rows = []
        for cid, samples in self.buffer.items():
            if samples:
                rows.append((now.isoformat(timespec="seconds"), int(now.timestamp()), self.core_count, cid, median(samples), governor))
        if rows:
            self.executemany("""
                INSERT INTO cpu_usage(timestamp, timestamp_ts, cpu_cores, cpu_core_id, cpu_usage, cpu_governor)
                VALUES (?, ?, ?, ?, ?, ?)
            """, rows, commit=True)
            self.buffer = {i: deque(maxlen=900) for i in range(self.core_count)}
            logging.info("Committed CPU usage for %d cores.", len(rows))

    def log_governor_event(self, old: str, new: str, decision: Decision) -> None:
        now = datetime.now().astimezone()
        self.execute("""
            INSERT INTO governor_events(timestamp, timestamp_ts, old_governor, new_governor, state, score, reason, decision_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (now.isoformat(timespec="seconds"), int(now.timestamp()), old, new, decision.state, decision.score, decision.reason, json.dumps(asdict(decision))), commit=True)

    def prune(self, days: int) -> None:
        cutoff = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp())
        self.execute("DELETE FROM cpu_usage WHERE timestamp_ts < ?", (cutoff,), commit=True)
        self.execute("DELETE FROM governor_events WHERE timestamp_ts < ?", (cutoff,), commit=True)


class PriceFetcher:
    def __init__(self, timeout: int):
        self.timeout = timeout
        self.session = requests.Session()
        self.backoff: dict[str, tuple[int, float]] = {}

    def can_try(self, key: str) -> bool:
        return time.time() >= self.backoff.get(key, (0, 0))[1]

    def success(self, key: str) -> None:
        self.backoff.pop(key, None)

    def fail(self, key: str) -> None:
        tries, _ = self.backoff.get(key, (0, 0))
        tries += 1
        delay = min(60 * (2 ** (tries - 1)), 3600)
        self.backoff[key] = (tries, time.time() + delay + random.uniform(0, delay * 0.10))

    def fetch(self, area: str, d: date) -> Optional[list[dict]]:
        key = d.isoformat()
        if not self.can_try(key):
            return None
        api_date = d.strftime("%Y/%m-%d")
        url = f"https://www.elprisetjustnu.se/api/v1/prices/{api_date}_{area}.json"
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            logging.warning("Price fetch failed for %s: %s", key, exc)
            self.fail(key)
            return None
        if not isinstance(data, list):
            self.fail(key)
            return None
        valid = [x for x in data if isinstance(x, dict) and all(k in x for k in ("SEK_per_kWh", "time_start", "time_end"))]
        if valid:
            self.success(key)
            return valid
        self.fail(key)
        return None


class CPUGovernor:
    @staticmethod
    def files() -> list[Path]:
        return [Path(p) for p in glob.glob("/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor")]

    @classmethod
    def current(cls) -> Optional[str]:
        for f in cls.files():
            try:
                return f.read_text(encoding="utf-8").strip()
            except OSError:
                continue
        return None

    @staticmethod
    def available_for(gov_file: Path) -> set[str]:
        try:
            return set(gov_file.with_name("scaling_available_governors").read_text(encoding="utf-8").split())
        except OSError:
            return set(PREFERRED_GOVERNORS)

    @classmethod
    def supported(cls) -> set[str]:
        supported = None
        for f in cls.files():
            available = cls.available_for(f)
            supported = available if supported is None else supported.intersection(available)
        return supported or set()

    @classmethod
    def closest_supported(cls, desired: str) -> str:
        supported = cls.supported()
        if not supported or desired in supported:
            return desired
        desired_p = GOVERNOR_PRIORITY.get(desired, 0)
        candidates = [g for g in supported if g in GOVERNOR_PRIORITY]
        if not candidates:
            return desired
        return sorted(candidates, key=lambda g: (abs(GOVERNOR_PRIORITY[g] - desired_p), -GOVERNOR_PRIORITY[g]))[0]

    @classmethod
    def set_all(cls, governor: str, dry_run: bool = False) -> bool:
        target = cls.closest_supported(governor)
        files = cls.files()
        if not files:
            logging.error("No scaling_governor files found.")
            return False
        if dry_run:
            logging.info("DRY-RUN would set governor to %s on %d CPU policies.", target, len(files))
            return True
        ok = 0
        for f in files:
            try:
                if f.read_text(encoding="utf-8").strip() != target:
                    f.write_text(target, encoding="utf-8")
                if f.read_text(encoding="utf-8").strip() == target:
                    ok += 1
            except OSError as exc:
                logging.error("Failed writing %s: %s", f, exc)
        logging.info("Governor write result: %d/%d CPU policies set to %s.", ok, len(files), target)
        return ok == len(files)


class MetricsCollector:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.smoothed_cpu: Optional[float] = None
        self.cpu_history = deque(maxlen=cfg.history_samples)

    def aggregate_cpu(self, values: list[float]) -> float:
        if not values:
            return 0.0
        if self.cfg.usage_method == "average":
            return mean(values)
        if self.cfg.usage_method == "max":
            return max(values)
        return median(values)

    def temperature(self) -> Optional[float]:
        if not self.cfg.thermal_enabled:
            return None
        try:
            temps = psutil.sensors_temperatures(fahrenheit=False)
        except Exception:
            return None
        candidates = []
        for entries in temps.values():
            for entry in entries:
                if entry.current is not None:
                    candidates.append(float(entry.current))
        return max(candidates) if candidates else None

    def collect(self, price_mgr: PriceManager, cpu_db: CPUHistory) -> Metrics:
        usage_list = psutil.cpu_percent(interval=1, percpu=True)
        for cid, usage in enumerate(usage_list):
            cpu_db.add_sample(cid, usage)
        cpu = self.aggregate_cpu(usage_list)
        self.smoothed_cpu = cpu if self.smoothed_cpu is None else self.cfg.smooth_factor * cpu + (1 - self.cfg.smooth_factor) * self.smoothed_cpu
        self.cpu_history.append(self.smoothed_cpu)
        sustained = sum(1 for x in self.cpu_history if x >= self.cfg.high_load_threshold) / len(self.cpu_history)
        try:
            load_1m = os.getloadavg()[0]
            cores = psutil.cpu_count(logical=True) or 1
            load_ratio = clamp(load_1m / cores)
        except OSError:
            load_ratio = 0.0
        try:
            iowait = float(getattr(psutil.cpu_times_percent(interval=None), "iowait", 0.0))
        except Exception:
            iowait = 0.0
        today = datetime.now().astimezone().date()
        current_price = price_mgr.current_price()
        rank = price_mgr.price_rank(current_price, today)
        future_rank = price_mgr.future_average_rank(today, self.cfg.lookahead_hours)
        trend = None if rank is None or future_rank is None else future_rank - rank
        return Metrics(datetime.now().astimezone().isoformat(timespec="seconds"), cpu, float(self.smoothed_cpu), load_ratio, iowait, sustained, self.temperature(), current_price, rank, future_rank, trend)


class DecisionEngine:
    POLICY = {
        WorkloadClass.IDLE: {PriceClass.CHEAP: "powersave", PriceClass.NORMAL: "powersave", PriceClass.EXPENSIVE: "powersave", PriceClass.UNKNOWN: "powersave"},
        WorkloadClass.LIGHT: {PriceClass.CHEAP: "conservative", PriceClass.NORMAL: "powersave", PriceClass.EXPENSIVE: "powersave", PriceClass.UNKNOWN: "powersave"},
        WorkloadClass.BALANCED: {PriceClass.CHEAP: "schedutil", PriceClass.NORMAL: "conservative", PriceClass.EXPENSIVE: "powersave", PriceClass.UNKNOWN: "conservative"},
        WorkloadClass.BURST: {PriceClass.CHEAP: "schedutil", PriceClass.NORMAL: "schedutil", PriceClass.EXPENSIVE: "conservative", PriceClass.UNKNOWN: "conservative"},
        WorkloadClass.SUSTAINED_HEAVY: {PriceClass.CHEAP: "performance", PriceClass.NORMAL: "schedutil", PriceClass.EXPENSIVE: "schedutil", PriceClass.UNKNOWN: "schedutil"},
        WorkloadClass.IO_BOUND: {PriceClass.CHEAP: "conservative", PriceClass.NORMAL: "conservative", PriceClass.EXPENSIVE: "powersave", PriceClass.UNKNOWN: "conservative"},
    }

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def one_level(self, governor: str, delta: int) -> str:
        p = GOVERNOR_PRIORITY.get(governor, 0)
        return PRIORITY_GOVERNOR[int(max(0, min(3, p + delta)))]

    def thermal_class(self, temp: Optional[float]) -> ThermalClass:
        if temp is None:
            return ThermalClass.UNKNOWN
        if temp >= self.cfg.critical_temp:
            return ThermalClass.CRITICAL
        if temp >= self.cfg.hot_temp:
            return ThermalClass.HOT
        if temp >= self.cfg.warm_temp:
            return ThermalClass.WARM
        return ThermalClass.NORMAL

    def price_class(self, rank: Optional[float]) -> PriceClass:
        if rank is None:
            return PriceClass.UNKNOWN
        if rank <= self.cfg.cheap_rank_max:
            return PriceClass.CHEAP
        if rank >= self.cfg.expensive_rank_min:
            return PriceClass.EXPENSIVE
        return PriceClass.NORMAL

    def workload_class(self, m: Metrics) -> WorkloadClass:
        if m.iowait_percent >= self.cfg.iowait_threshold and m.smoothed_cpu >= 25:
            return WorkloadClass.IO_BOUND
        if m.smoothed_cpu <= self.cfg.idle_cpu_threshold and m.loadavg_1m_ratio < 0.25:
            return WorkloadClass.IDLE
        if m.smoothed_cpu >= self.cfg.high_load_threshold and m.sustained_load_ratio >= 0.50:
            return WorkloadClass.SUSTAINED_HEAVY
        if m.smoothed_cpu >= self.cfg.burst_cpu_threshold and m.sustained_load_ratio < 0.50:
            return WorkloadClass.BURST
        if m.smoothed_cpu < 35 and m.loadavg_1m_ratio < 0.40:
            return WorkloadClass.LIGHT
        return WorkloadClass.BALANCED

    def score_to_delta(self, score: float) -> int:
        if score >= 78:
            return 1
        if score <= 22:
            return -1
        return 0

    def decide(self, current_governor: str, m: Metrics) -> Decision:
        t_class = self.thermal_class(m.cpu_temp)
        p_class = self.price_class(m.price_rank)
        w_class = self.workload_class(m)
        thermal_penalty = 0.0
        if m.cpu_temp is not None and self.cfg.thermal_enabled:
            thermal_penalty = clamp((m.cpu_temp - self.cfg.warm_temp) / max(1.0, self.cfg.critical_temp - self.cfg.warm_temp))
        if t_class == ThermalClass.CRITICAL:
            return Decision("powersave", "thermal_protect", 0.0, w_class.value, p_class.value, t_class.value, 0.0, 0.0, 0.0, 0.0, round(thermal_penalty, 3), "powersave", "Critical CPU temperature overrides price and workload.", True)

        load_score = clamp((m.smoothed_cpu / 100) * 0.70 + m.loadavg_1m_ratio * 0.20 + clamp(m.iowait_percent / 100) * 0.10)
        urgency_score = clamp(m.sustained_load_ratio)
        cheap_now_score = 0.50 if m.price_rank is None else clamp(1.0 - m.price_rank)
        lookahead_score = 0.0 if m.price_trend is None else clamp(m.price_trend)
        raw = load_score * self.cfg.cpu_weight + urgency_score * self.cfg.urgency_weight + cheap_now_score * self.cfg.cheap_now_weight + lookahead_score * self.cfg.lookahead_weight - thermal_penalty * self.cfg.thermal_weight
        score = clamp(raw) * 100.0
        baseline = self.POLICY[w_class][p_class]
        governor = self.one_level(baseline, self.score_to_delta(score))

        if t_class == ThermalClass.HOT:
            governor = self.one_level(governor, -2)
        elif t_class == ThermalClass.WARM:
            governor = self.one_level(governor, -1)
        if m.price_trend is not None:
            if m.price_trend >= self.cfg.lookahead_delta and p_class != PriceClass.EXPENSIVE and w_class in (WorkloadClass.BALANCED, WorkloadClass.BURST, WorkloadClass.SUSTAINED_HEAVY):
                governor = self.one_level(governor, 1)
            elif m.price_trend <= -self.cfg.lookahead_delta and w_class not in (WorkloadClass.SUSTAINED_HEAVY, WorkloadClass.BURST):
                governor = self.one_level(governor, -1)
        if p_class == PriceClass.EXPENSIVE and w_class not in (WorkloadClass.SUSTAINED_HEAVY, WorkloadClass.BURST):
            governor = self.one_level(governor, -1)

        governor = CPUGovernor.closest_supported(governor)
        reason = self.reason(w_class, p_class, t_class, score, m)
        return Decision(governor, f"{w_class.value}_{p_class.value}", round(score, 1), w_class.value, p_class.value, t_class.value, round(load_score, 3), round(urgency_score, 3), round(cheap_now_score, 3), round(lookahead_score, 3), round(thermal_penalty, 3), baseline, reason)

    def reason(self, workload: WorkloadClass, price: PriceClass, thermal: ThermalClass, score: float, m: Metrics) -> str:
        parts = [f"{workload.value.replace('_', ' ')} workload", f"{price.value} price", f"score {score:.1f}"]
        if m.price_trend is not None:
            if m.price_trend >= self.cfg.lookahead_delta:
                parts.append("prices are rising in the lookahead window")
            elif m.price_trend <= -self.cfg.lookahead_delta:
                parts.append("prices are falling in the lookahead window")
        if thermal in (ThermalClass.WARM, ThermalClass.HOT):
            parts.append(f"CPU is {thermal.value}")
        return "; ".join(parts) + "."


class TransitionManager:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.last_change_ts: Optional[float] = None
        self.last_upscale_ts: Optional[float] = None
        self.last_score: Optional[float] = None

    def allow(self, current: str, decision: Decision) -> tuple[str, str]:
        desired = decision.governor
        if desired == current or decision.safety_override:
            return desired, "allowed"
        now = time.time()
        cur_p = GOVERNOR_PRIORITY.get(current, 0)
        new_p = GOVERNOR_PRIORITY.get(desired, cur_p)
        if self.last_change_ts and now - self.last_change_ts < self.cfg.min_hold_sec and new_p < cur_p:
            return current, "minimum hold active"
        if self.last_upscale_ts and new_p < cur_p and now - self.last_upscale_ts < self.cfg.upscale_cooldown_sec:
            return current, "upscale cooldown active"
        if self.last_score is not None and abs(decision.score - self.last_score) < self.cfg.score_margin and new_p != cur_p:
            return current, "score margin not large enough"
        return desired, "allowed"

    def committed(self, old: str, new: str, decision: Decision) -> None:
        now = time.time()
        if old != new:
            self.last_change_ts = now
            if GOVERNOR_PRIORITY.get(new, 0) > GOVERNOR_PRIORITY.get(old, 0):
                self.last_upscale_ts = now
        self.last_score = decision.score


def setup_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level, logging.INFO), format="%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


def fetch_needed_prices(price_mgr: PriceManager, fetcher: PriceFetcher, cfg: Config) -> None:
    now = datetime.now().astimezone()
    dates = [now.date()]
    if now.hour >= 13:
        dates.append(now.date() + timedelta(days=1))
    for d in dates:
        if price_mgr.has_date(d):
            continue
        data = fetcher.fetch(cfg.area, d)
        if data:
            price_mgr.insert_bulk(data)
            logging.info("Inserted %d price rows for %s.", len(data), d.isoformat())


def table(title: str, rows, fields: list[str]) -> None:
    print(f"\n{title}")
    t = PrettyTable()
    t.field_names = fields
    t.align = "l"
    if not rows:
        print("No rows found.")
        return
    for row in rows:
        t.add_row([row[field] if field in row.keys() else "" for field in fields])
    print(t)


def report(price_mgr: PriceManager, cpu_db: CPUHistory) -> None:
    price_rows = price_mgr.execute("""
        SELECT time_start, time_end, ROUND(SEK_per_kWh, 4) AS SEK_per_kWh
        FROM prices
        ORDER BY start_ts DESC
        LIMIT 48
    """)
    table("Latest prices", price_rows, ["time_start", "time_end", "SEK_per_kWh"])

    event_rows = cpu_db.execute("""
        SELECT timestamp, old_governor, new_governor, state, ROUND(score, 1) AS score, reason
        FROM governor_events
        ORDER BY timestamp_ts DESC
        LIMIT 50
    """)
    table("Governor events", event_rows, ["timestamp", "old_governor", "new_governor", "state", "score", "reason"])

    usage_rows = cpu_db.execute("""
        SELECT timestamp, cpu_core_id, ROUND(cpu_usage, 1) AS cpu_usage, cpu_governor
        FROM cpu_usage
        ORDER BY timestamp_ts DESC, cpu_core_id ASC
        LIMIT 64
    """)
    table("CPU usage samples", usage_rows, ["timestamp", "cpu_core_id", "cpu_usage", "cpu_governor"])


def debug_dump(price_mgr: PriceManager, cpu_db: CPUHistory) -> None:
    print("Latest prices:")
    for row in price_mgr.execute("SELECT * FROM prices ORDER BY start_ts DESC LIMIT 48"):
        print(dict(row))
    print("Latest governor events:")
    for row in cpu_db.execute("SELECT * FROM governor_events ORDER BY timestamp_ts DESC LIMIT 50"):
        print(dict(row))


def check_config(cfg: Config) -> int:
    problems = []
    if cfg.cheap_rank_max >= cfg.expensive_rank_min:
        problems.append("CHEAP_RANK_MAX must be lower than EXPENSIVE_RANK_MIN")
    if cfg.warm_temp >= cfg.hot_temp or cfg.hot_temp >= cfg.critical_temp:
        problems.append("Thermal temperatures must be WARM_TEMP < HOT_TEMP < CRITICAL_TEMP")
    if not CPUGovernor.files():
        problems.append("No scaling_governor files found")

    t = PrettyTable()
    t.field_names = ["Check", "Result"]
    t.align = "l"
    t.add_row(["Area", cfg.area])
    t.add_row(["Dry run", str(cfg.dry_run)])
    t.add_row(["Supported governors", ", ".join(sorted(CPUGovernor.supported())) or "unknown"])
    t.add_row(["Governor files", len(CPUGovernor.files())])
    print(t)

    if problems:
        for p in problems:
            print(f"ERROR: {p}")
        return 1
    print("Config looks sane.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="PowerNap formulaic CPU governor controller")
    parser.add_argument("--config", default=os.environ.get("POWERNAP_CONFIG"), help="Path to powernap.conf")
    parser.add_argument("--report", action="store_true", help="Show PrettyTable presentation report and exit")
    parser.add_argument("--debug", action="store_true", help="Show raw diagnostic database output and exit")
    parser.add_argument("--dry-run", action="store_true", help="Calculate decisions without writing governors")
    parser.add_argument("--check-config", action="store_true", help="Validate config and environment")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    config_path = Path(args.config) if args.config else script_dir / DEFAULT_CONFIG_NAME
    cfg_preview = load_config(config_path, args.dry_run)
    setup_logging(cfg_preview.log_level)
    cfg = load_config(config_path, args.dry_run)

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    if args.check_config:
        return check_config(cfg)

    db_dir = Path(cfg.database_dir)
    if not db_dir.is_absolute():
        db_dir = script_dir / db_dir
    price_mgr = PriceManager(db_dir / "prices.db")
    cpu_db = CPUHistory(db_dir / "cpu.db", psutil.cpu_count(logical=True) or 1)
    fetcher = PriceFetcher(cfg.request_timeout_sec)
    collector = MetricsCollector(cfg)
    engine = DecisionEngine(cfg)
    transition = TransitionManager(cfg)

    try:
        if args.report:
            report(price_mgr, cpu_db)
            return 0
        if args.debug:
            debug_dump(price_mgr, cpu_db)
            return 0

        current_governor = CPUGovernor.current() or "unknown"
        last_commit_bucket = None
        last_cleanup_date = datetime.now().date()
        logging.info("%s v2 started. governor=%s dry_run=%s", APP_NAME, current_governor, cfg.dry_run)

        while not STOP_REQUESTED:
            cycle_start = time.monotonic()
            fetch_needed_prices(price_mgr, fetcher, cfg)
            metrics = collector.collect(price_mgr, cpu_db)
            decision = engine.decide(current_governor, metrics)
            target, gate_reason = transition.allow(current_governor, decision)

            if target != decision.governor:
                logging.info("Transition held: wanted=%s held=%s reason=%s decision=%s", decision.governor, target, gate_reason, decision.reason)

            if target != current_governor:
                old = current_governor
                if CPUGovernor.set_all(target, dry_run=cfg.dry_run):
                    current_governor = target
                    transition.committed(old, current_governor, decision)
                    cpu_db.log_governor_event(old, current_governor, decision)
                    logging.info("Governor changed: %s -> %s | %s", old, current_governor, decision.reason)
            else:
                transition.committed(current_governor, current_governor, decision)

            now = datetime.now().astimezone()
            if cfg.commit_interval_minutes > 0:
                bucket = int(now.timestamp() // (cfg.commit_interval_minutes * 60))
                if bucket != last_commit_bucket:
                    cpu_db.commit_usage(current_governor)
                    last_commit_bucket = bucket
            if cfg.data_retention_enabled and now.date() != last_cleanup_date:
                price_mgr.prune(cfg.data_retention_days)
                cpu_db.prune(cfg.data_retention_days)
                last_cleanup_date = now.date()

            elapsed = time.monotonic() - cycle_start
            time.sleep(max(0.1, cfg.sleep_interval - elapsed))

        cpu_db.commit_usage(current_governor)
        logging.info("%s stopped cleanly.", APP_NAME)
        return 0
    finally:
        price_mgr.close()
        cpu_db.close()


if __name__ == "__main__":
    raise SystemExit(main())
