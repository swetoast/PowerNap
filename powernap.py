#!/usr/bin/env python3
"""
PowerNap - electricity-price-aware CPU governor controller.

Design goals:
- Prefer stability over rapid governor flapping.
- Store electricity-price intervals with epoch timestamps for reliable lookup.
- Keep the runtime small: stdlib + requests + psutil only.
- Be systemd-friendly and safe to stop with SIGTERM/SIGINT.
"""
from __future__ import annotations

import configparser
import glob
import logging
import os
import random
import signal
import sqlite3
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean, median
from typing import Iterable, Optional

import psutil
import requests

APP_NAME = "PowerNap"
DEFAULT_CONFIG_NAME = "powernap.conf"
GOVERNOR_PRIORITY = {
    "powersave": 0,
    "conservative": 1,
    "schedutil": 2,
    "performance": 3,
}
PREFERRED_GOVERNORS = ("performance", "schedutil", "conservative", "powersave")
STOP_REQUESTED = False


def request_stop(signum, _frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    logging.info("Received signal %s; shutting down cleanly after current cycle.", signum)


@dataclass(frozen=True)
class Config:
    high_cost: float = 1.0
    mid_cost: float = 0.5
    low_cost: float = 0.1
    area: str = "SE3"
    sleep_interval: int = 5
    data_retention_days: int = 30
    data_retention_enabled: bool = True
    commit_interval_minutes: int = 5
    usage_method: str = "median"
    perf_enter_util: float = 85.0
    perf_exit_util: float = 70.0
    psave_enter_util: float = 30.0
    psave_exit_util: float = 40.0
    smooth_factor: float = 0.30
    cooldown_sec: int = 15
    request_timeout_sec: int = 10
    log_level: str = "INFO"
    database_dir: str = "."

    @property
    def thresholds(self) -> tuple[float, float, float]:
        return self.high_cost, self.mid_cost, self.low_cost


def _strip_inline_comment(value: str) -> str:
    return value.split("#", 1)[0].strip()


def load_config(config_path: Path) -> Config:
    parser = configparser.ConfigParser()
    parser.optionxform = str
    parser.read(config_path)

    def get(section: str, option: str, default, cast):
        try:
            raw = parser.get(section, option)
        except (configparser.NoSectionError, configparser.NoOptionError):
            return default
        if cast is bool:
            return parser.getboolean(section, option)
        if cast is str:
            return _strip_inline_comment(raw)
        return cast(_strip_inline_comment(raw))

    cfg = Config(
        high_cost=get("CostConstants", "HIGH_COST", Config.high_cost, float),
        mid_cost=get("CostConstants", "MID_COST", Config.mid_cost, float),
        low_cost=get("CostConstants", "LOW_COST", Config.low_cost, float),
        area=get("AreaCode", "AREA", Config.area, str),
        sleep_interval=max(1, get("SleepInterval", "INTERVAL", Config.sleep_interval, int)),
        data_retention_days=max(1, get("DataRetention", "DAYS", Config.data_retention_days, int)),
        data_retention_enabled=get("DataRetention", "ENABLED", Config.data_retention_enabled, bool),
        commit_interval_minutes=max(0, get("CommitInterval", "INTERVAL", Config.commit_interval_minutes, int)),
        usage_method=get("UsageCalculation", "METHOD", Config.usage_method, str).lower(),
        perf_enter_util=get("GovernorLogic", "PERF_ENTER_UTIL", Config.perf_enter_util, float),
        perf_exit_util=get("GovernorLogic", "PERF_EXIT_UTIL", Config.perf_exit_util, float),
        psave_enter_util=get("GovernorLogic", "PSAVE_ENTER_UTIL", Config.psave_enter_util, float),
        psave_exit_util=get("GovernorLogic", "PSAVE_EXIT_UTIL", Config.psave_exit_util, float),
        smooth_factor=get("GovernorLogic", "SMOOTH_FACTOR", Config.smooth_factor, float),
        cooldown_sec=max(0, get("GovernorLogic", "COOLDOWN_SEC", Config.cooldown_sec, int)),
        request_timeout_sec=max(2, get("Network", "REQUEST_TIMEOUT_SEC", Config.request_timeout_sec, int)),
        log_level=get("Logging", "LEVEL", Config.log_level, str).upper(),
        database_dir=get("Storage", "DATABASE_DIR", Config.database_dir, str),
    )

    if cfg.usage_method not in ("average", "median"):
        logging.warning("Unknown UsageCalculation.METHOD=%r; using median for runtime decisions.", cfg.usage_method)
        cfg = Config(**{**cfg.__dict__, "usage_method": "median"})
    if not (cfg.high_cost > cfg.mid_cost > cfg.low_cost):
        logging.warning(
            "Cost thresholds should be HIGH_COST > MID_COST > LOW_COST; got %.3f > %.3f > %.3f.",
            cfg.high_cost,
            cfg.mid_cost,
            cfg.low_cost,
        )
    if not (0 < cfg.smooth_factor <= 1):
        logging.warning("SMOOTH_FACTOR must be > 0 and <= 1; using 0.30.")
        cfg = Config(**{**cfg.__dict__, "smooth_factor": 0.30})
    return cfg


def secure_owner_rw(path: Path) -> None:
    try:
        if path.exists():
            path.chmod(0o600)
    except OSError as exc:
        logging.debug("Could not chmod %s: %s", path, exc)


def parse_price_time(value: str) -> int:
    # API values are expected to be ISO-8601, often with +01:00/+02:00 offsets.
    normalized = value.replace("Z", "+00:00")
    return int(datetime.fromisoformat(normalized).timestamp())


class DatabaseManager:
    def __init__(self, db_file: Path):
        self.db_file = db_file
        self.db_file.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_file), detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
        self.conn.row_factory = sqlite3.Row
        self._set_pragmas()
        secure_owner_rw(db_file)

    def _set_pragmas(self) -> None:
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.conn.execute("PRAGMA temp_store=MEMORY;")
        self.conn.execute("PRAGMA foreign_keys=ON;")

    def execute(self, query: str, params: Iterable = (), commit: bool = False):
        try:
            cur = self.conn.execute(query, tuple(params))
            if commit:
                self.conn.commit()
            return cur.fetchall()
        except sqlite3.Error as exc:
            logging.error("Database error in %s: %s", self.db_file, exc)
            logging.debug("Failed query: %s", query)
            return []

    def executemany(self, query: str, seq_of_params: Iterable[Iterable], commit: bool = False):
        try:
            cur = self.conn.executemany(query, seq_of_params)
            if commit:
                self.conn.commit()
            return cur.fetchall()
        except sqlite3.Error as exc:
            logging.error("Database batch error in %s: %s", self.db_file, exc)
            logging.debug("Failed query: %s", query)
            return []

    def close(self) -> None:
        self.conn.close()


class PriceManager(DatabaseManager):
    def __init__(self, db_file: Path):
        super().__init__(db_file)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS prices (
                id INTEGER PRIMARY KEY,
                SEK_per_kWh REAL NOT NULL,
                time_start TEXT NOT NULL,
                time_end TEXT NOT NULL,
                start_ts INTEGER,
                end_ts INTEGER,
                UNIQUE(time_start, time_end)
            )
            """
        )
        self._ensure_epoch_columns()
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_prices_epoch ON prices(start_ts, end_ts);")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_prices_text_time ON prices(time_start, time_end);")
        self.conn.commit()

    def _ensure_epoch_columns(self) -> None:
        columns = {row["name"] for row in self.conn.execute("PRAGMA table_info(prices)")}
        if "start_ts" not in columns:
            self.conn.execute("ALTER TABLE prices ADD COLUMN start_ts INTEGER")
        if "end_ts" not in columns:
            self.conn.execute("ALTER TABLE prices ADD COLUMN end_ts INTEGER")
        rows = self.conn.execute("SELECT id, time_start, time_end FROM prices WHERE start_ts IS NULL OR end_ts IS NULL").fetchall()
        for row in rows:
            try:
                self.conn.execute(
                    "UPDATE prices SET start_ts=?, end_ts=? WHERE id=?",
                    (parse_price_time(row["time_start"]), parse_price_time(row["time_end"]), row["id"]),
                )
            except ValueError:
                logging.warning("Could not parse stored price interval id=%s", row["id"])
        self.conn.commit()

    def insert_bulk(self, data_list: list[tuple[float, str, str]]) -> None:
        if not data_list:
            return
        rows = []
        for price, start, end in data_list:
            try:
                rows.append((float(price), start, end, parse_price_time(start), parse_price_time(end)))
            except (TypeError, ValueError) as exc:
                logging.warning("Skipping invalid price row %r: %s", (price, start, end), exc)
        self.executemany(
            """
            INSERT OR IGNORE INTO prices(SEK_per_kWh, time_start, time_end, start_ts, end_ts)
            VALUES (?,?,?,?,?)
            """,
            rows,
            commit=True,
        )

    def get_current_price(self, now_ts: Optional[int] = None) -> Optional[float]:
        now_ts = int(time.time()) if now_ts is None else now_ts
        rows = self.execute(
            """
            SELECT SEK_per_kWh
            FROM prices
            WHERE start_ts <= ? AND end_ts > ?
            ORDER BY start_ts DESC
            LIMIT 1
            """,
            (now_ts, now_ts),
        )
        return float(rows[0]["SEK_per_kWh"]) if rows else None

    def has_data_for_date(self, date_prefix: str) -> bool:
        rows = self.execute("SELECT 1 FROM prices WHERE time_start LIKE ? LIMIT 1", (date_prefix + "%",))
        return bool(rows)

    def remove_old_data(self, days: int) -> None:
        cutoff = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp())
        self.execute("DELETE FROM prices WHERE end_ts < ?", (cutoff,), commit=True)


class CPUManager(DatabaseManager):
    def __init__(self, db_file: Path, core_count: int):
        super().__init__(db_file)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cpu_usage (
                id INTEGER PRIMARY KEY,
                timestamp TEXT NOT NULL,
                timestamp_ts INTEGER NOT NULL,
                cpu_cores INTEGER NOT NULL,
                cpu_core_id INTEGER NOT NULL,
                cpu_usage REAL NOT NULL,
                cpu_governor TEXT NOT NULL
            )
            """
        )
        self._ensure_epoch_column()
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_cpu_timestamp_ts ON cpu_usage(timestamp_ts);")
        self.conn.commit()
        self.core_count = core_count
        self.cpu_data: dict[int, deque[float]] = {i: deque(maxlen=900) for i in range(core_count)}

    def _ensure_epoch_column(self) -> None:
        columns = {row["name"] for row in self.conn.execute("PRAGMA table_info(cpu_usage)")}
        if "timestamp_ts" not in columns:
            self.conn.execute("ALTER TABLE cpu_usage ADD COLUMN timestamp_ts INTEGER")
            rows = self.conn.execute("SELECT id, timestamp FROM cpu_usage WHERE timestamp_ts IS NULL").fetchall()
            for row in rows:
                try:
                    ts = int(datetime.fromisoformat(str(row["timestamp"]).replace("Z", "+00:00")).timestamp())
                except ValueError:
                    ts = int(time.time())
                self.conn.execute("UPDATE cpu_usage SET timestamp_ts=? WHERE id=?", (ts, row["id"]))
            self.conn.commit()

    def insert_temp(self, core_id: int, usage: float) -> None:
        if core_id in self.cpu_data:
            self.cpu_data[core_id].append(float(usage))

    def commit_data(self, current_governor: str) -> None:
        now = datetime.now().astimezone()
        now_iso = now.isoformat(timespec="seconds")
        now_ts = int(now.timestamp())
        batch = []
        for cid, dq in self.cpu_data.items():
            if dq:
                batch.append((now_iso, now_ts, self.core_count, cid, median(dq), current_governor))
        if batch:
            self.executemany(
                """
                INSERT INTO cpu_usage(timestamp, timestamp_ts, cpu_cores, cpu_core_id, cpu_usage, cpu_governor)
                VALUES (?,?,?,?,?,?)
                """,
                batch,
                commit=True,
            )
            self.cpu_data = {i: deque(maxlen=900) for i in range(self.core_count)}
            logging.info("Committed CPU usage for %d cores.", len(batch))

    def remove_old_data(self, days: int) -> None:
        cutoff = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp())
        self.execute("DELETE FROM cpu_usage WHERE timestamp_ts < ?", (cutoff,), commit=True)


class PriceFetcher:
    def __init__(self, timeout_sec: int):
        self.timeout_sec = timeout_sec
        self.session = requests.Session()

    def fetch(self, area_code: str, date_str: str) -> Optional[list[dict]]:
        url = f"https://www.elprisetjustnu.se/api/v1/prices/{date_str}_{area_code}.json"
        try:
            response = self.session.get(url, timeout=self.timeout_sec)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            logging.warning("Price API request failed for %s/%s: %s", area_code, date_str, exc)
            return None
        except ValueError as exc:
            logging.warning("Could not parse price API response for %s/%s: %s", area_code, date_str, exc)
            return None

        if not isinstance(data, list):
            logging.warning("Unexpected price API response type: %s", type(data).__name__)
            return None
        valid = []
        for item in data:
            if not isinstance(item, dict):
                continue
            if not all(key in item for key in ("SEK_per_kWh", "time_start", "time_end")):
                continue
            if not isinstance(item["SEK_per_kWh"], (int, float)):
                continue
            valid.append(item)
        if not valid:
            logging.warning("No valid price rows returned for %s/%s.", area_code, date_str)
            return None
        return valid


class Backoff:
    def __init__(self, base_sec: int = 60, max_sec: int = 3600):
        self.base_sec = base_sec
        self.max_sec = max_sec
        self.retry_count = 0
        self.next_retry_ts = 0.0

    def can_try(self) -> bool:
        return time.time() >= self.next_retry_ts

    def success(self) -> None:
        self.retry_count = 0
        self.next_retry_ts = 0.0

    def fail(self) -> int:
        self.retry_count += 1
        delay = min(self.base_sec * (2 ** (self.retry_count - 1)), self.max_sec)
        jitter = random.uniform(0, delay * 0.10)
        self.next_retry_ts = time.time() + delay + jitter
        return int(delay + jitter)


class CPUGovernor:
    @staticmethod
    def governor_files() -> list[Path]:
        return [Path(p) for p in glob.glob("/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor")]

    @staticmethod
    def available_governors_for(gov_file: Path) -> set[str]:
        available_file = gov_file.with_name("scaling_available_governors")
        try:
            return set(available_file.read_text(encoding="utf-8").split())
        except OSError:
            return set(PREFERRED_GOVERNORS)

    @classmethod
    def current(cls) -> Optional[str]:
        for gov_file in cls.governor_files():
            try:
                return gov_file.read_text(encoding="utf-8").strip()
            except OSError:
                continue
        return None

    @classmethod
    def supported_governors(cls) -> set[str]:
        supported: Optional[set[str]] = None
        for gov_file in cls.governor_files():
            available = cls.available_governors_for(gov_file)
            supported = available if supported is None else supported.intersection(available)
        return supported or set()

    @classmethod
    def closest_supported(cls, desired: str) -> Optional[str]:
        supported = cls.supported_governors()
        if not supported:
            return desired
        if desired in supported:
            return desired
        desired_priority = GOVERNOR_PRIORITY.get(desired, 0)
        candidates = sorted(
            (g for g in supported if g in GOVERNOR_PRIORITY),
            key=lambda g: (abs(GOVERNOR_PRIORITY[g] - desired_priority), -GOVERNOR_PRIORITY[g]),
        )
        return candidates[0] if candidates else None

    @classmethod
    def set_all(cls, governor: str) -> bool:
        target = cls.closest_supported(governor)
        if not target:
            logging.error("No supported CPU governors found.")
            return False
        if target != governor:
            logging.info("Governor %r is unavailable; using closest supported governor %r.", governor, target)
        files = cls.governor_files()
        if not files:
            logging.error("No scaling_governor files found. Is CPU frequency scaling available?")
            return False
        ok = True
        for gov_file in files:
            try:
                gov_file.write_text(target, encoding="utf-8")
            except OSError as exc:
                logging.error("Failed to set %s to %r: %s", gov_file, target, exc)
                ok = False
        return ok


def aggregate_usage(values: list[float], method: str) -> float:
    if not values:
        return 0.0
    if method == "average":
        return mean(values)
    return median(values)


def choose_governor(current: str, usage: Optional[float], cost: Optional[float], cfg: Config) -> str:
    if cost is None:
        desired = "powersave"
    elif usage is None:
        desired = current if current in GOVERNOR_PRIORITY else "powersave"
    elif usage >= cfg.perf_enter_util and cost < cfg.low_cost:
        desired = "performance"
    elif usage >= 70 and cost < cfg.mid_cost:
        desired = "schedutil"
    elif usage <= cfg.psave_enter_util and cost > cfg.high_cost:
        desired = "powersave"
    elif usage <= 50 and cost >= cfg.mid_cost:
        desired = "powersave"
    elif usage < 70 and cost < cfg.mid_cost:
        desired = "conservative"
    else:
        desired = "powersave"

    if current == "performance" and desired != "performance" and usage is not None and usage >= cfg.perf_exit_util:
        return "performance"
    if current == "powersave" and desired != "powersave" and usage is not None and usage <= cfg.psave_exit_util:
        return "powersave"
    return desired


def fetch_prices_if_needed(price_mgr: PriceManager, fetcher: PriceFetcher, cfg: Config, backoffs: dict[str, Backoff]) -> None:
    now = datetime.now().astimezone()
    dates_to_fetch = [now.date()]
    if now.hour >= 13:
        dates_to_fetch.append(now.date() + timedelta(days=1))

    for day in dates_to_fetch:
        api_date = day.strftime("%Y/%m-%d")
        text_prefix = day.isoformat()
        if price_mgr.has_data_for_date(text_prefix):
            continue
        backoff = backoffs.setdefault(text_prefix, Backoff())
        if not backoff.can_try():
            continue
        data = fetcher.fetch(cfg.area, api_date)
        if data:
            rows = [(item["SEK_per_kWh"], item["time_start"], item["time_end"]) for item in data]
            price_mgr.insert_bulk(rows)
            backoff.success()
            logging.info("Inserted %d electricity price rows for %s.", len(rows), text_prefix)
        else:
            delay = backoff.fail()
            logging.warning("Price fetch for %s failed; retry delayed by about %s seconds.", text_prefix, delay)


def debug_dump(price_mgr: PriceManager, cpu_mgr: CPUManager) -> None:
    print("Prices database:")
    for row in price_mgr.execute("SELECT * FROM prices ORDER BY start_ts DESC"):
        print(dict(row))
    print("CPU usage database:")
    for row in cpu_mgr.execute("SELECT * FROM cpu_usage ORDER BY timestamp_ts DESC"):
        print(dict(row))


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    config_path = Path(os.environ.get("POWERNAP_CONFIG", script_dir / DEFAULT_CONFIG_NAME))

    cfg_preview = load_config(config_path)
    setup_logging(cfg_preview.log_level)
    cfg = load_config(config_path)
    secure_owner_rw(config_path)

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    db_dir = Path(cfg.database_dir)
    if not db_dir.is_absolute():
        db_dir = script_dir / db_dir

    price_mgr = PriceManager(db_dir / "prices.db")
    cpu_mgr = CPUManager(db_dir / "cpu.db", psutil.cpu_count(logical=True) or 1)
    fetcher = PriceFetcher(timeout_sec=cfg.request_timeout_sec)
    backoffs: dict[str, Backoff] = {}

    try:
        if "-debug" in sys.argv or "--debug" in sys.argv:
            debug_dump(price_mgr, cpu_mgr)
            return 0

        current_governor = CPUGovernor.current() or "unknown"
        smoothed_usage: Optional[float] = None
        last_upscale_ts: Optional[float] = None
        last_commit_bucket: Optional[int] = None
        last_cleanup_date = datetime.now().date()

        logging.info("%s started. Current governor: %s", APP_NAME, current_governor)

        while not STOP_REQUESTED:
            cycle_start = time.monotonic()
            now = datetime.now().astimezone()

            fetch_prices_if_needed(price_mgr, fetcher, cfg, backoffs)
            current_price = price_mgr.get_current_price()

            try:
                usage_list = psutil.cpu_percent(interval=1, percpu=True)
            except Exception as exc:
                logging.error("psutil.cpu_percent failed: %s", exc)
                usage_list = []

            for cid, usage in enumerate(usage_list):
                cpu_mgr.insert_temp(cid, usage)

            instantaneous_usage = aggregate_usage(usage_list, cfg.usage_method) if usage_list else None
            if instantaneous_usage is not None:
                smoothed_usage = instantaneous_usage if smoothed_usage is None else (
                    cfg.smooth_factor * instantaneous_usage + (1 - cfg.smooth_factor) * smoothed_usage
                )

            desired_governor = choose_governor(current_governor, smoothed_usage, current_price, cfg)
            desired_governor = CPUGovernor.closest_supported(desired_governor) or desired_governor

            if current_governor in GOVERNOR_PRIORITY and desired_governor in GOVERNOR_PRIORITY:
                if GOVERNOR_PRIORITY[desired_governor] < GOVERNOR_PRIORITY[current_governor]:
                    if last_upscale_ts and (time.time() - last_upscale_ts) < cfg.cooldown_sec:
                        desired_governor = current_governor
                elif GOVERNOR_PRIORITY[desired_governor] > GOVERNOR_PRIORITY[current_governor]:
                    last_upscale_ts = time.time()

            if desired_governor and desired_governor != current_governor:
                if CPUGovernor.set_all(desired_governor):
                    logging.info(
                        "Governor changed: %s -> %s | usage=%.1f%% | price=%s SEK/kWh",
                        current_governor,
                        desired_governor,
                        smoothed_usage if smoothed_usage is not None else -1,
                        f"{current_price:.3f}" if current_price is not None else "unknown",
                    )
                    current_governor = desired_governor

            if cfg.commit_interval_minutes > 0:
                bucket = int(now.timestamp() // (cfg.commit_interval_minutes * 60))
                if bucket != last_commit_bucket:
                    cpu_mgr.commit_data(current_governor)
                    last_commit_bucket = bucket

            if cfg.data_retention_enabled and now.date() != last_cleanup_date:
                price_mgr.remove_old_data(cfg.data_retention_days)
                cpu_mgr.remove_old_data(cfg.data_retention_days)
                last_cleanup_date = now.date()
                logging.info("Purged data older than %d days.", cfg.data_retention_days)

            elapsed = time.monotonic() - cycle_start
            sleep_for = max(0.1, cfg.sleep_interval - elapsed)
            time.sleep(sleep_for)

        cpu_mgr.commit_data(current_governor)
        logging.info("%s stopped cleanly.", APP_NAME)
        return 0
    finally:
        price_mgr.close()
        cpu_mgr.close()


if __name__ == "__main__":
    raise SystemExit(main())
