from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


class ConfigError(ValueError):
    """Raised when configuration cannot be parsed or validated."""


@dataclass(frozen=True)
class Config:
    area: str = "SE3"
    timezone: str = "Europe/Stockholm"
    price_provider: str = "elpris_eu"
    fallback_provider: str = "elprisetjustnu"
    request_timeout_sec: float = 5.0
    price_cache_hours: float = 36.0
    lookahead_hours: int = 3
    sample_interval_sec: float = 5.0
    history_samples: int = 24
    busy_core_threshold: float = 60.0
    warm_temp_c: float = 60.0
    hot_temp_c: float = 75.0
    critical_temp_c: float = 90.0
    thermal_recovery_c: float = 5.0
    promote_samples: int = 2
    relax_seconds: float = 60.0
    minimum_residence_seconds: float = 60.0
    database_path: str = "powernap.db"
    retention_days: int = 30
    sample_retention_days: int = 7
    dry_run: bool = True
    manage_cpu: bool = True
    manage_nvidia: bool = True
    manage_amdgpu: bool = True
    manage_powercap: bool = False
    external_change_policy: str = "yield"
    restore_on_exit: bool = True
    protected_processes: tuple[str, ...] = ()


def load_config(path: Path | None, dry_run: bool = False) -> Config:
    parser = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    if path is not None:
        if not path.exists():
            raise ConfigError(f"Configuration file does not exist: {path}")
        try:
            with path.open(encoding="utf-8") as stream:
                parser.read_file(stream)
        except (OSError, configparser.Error) as exc:
            raise ConfigError(f"Unable to read configuration {path}: {exc}") from exc

    def value(section: str, key: str, default, cast):
        if not parser.has_option(section, key):
            return default
        raw = parser.get(section, key).strip()
        try:
            if cast is bool:
                return parser.getboolean(section, key)
            if cast is tuple:
                return tuple(dict.fromkeys(x.strip() for x in raw.split(",") if x.strip()))
            return cast(raw)
        except (ValueError, configparser.Error) as exc:
            raise ConfigError(f"Invalid [{section}] {key}: {raw!r}") from exc

    cfg = Config(
        area=value("price", "area", Config.area, str).upper(),
        timezone=value("price", "timezone", Config.timezone, str),
        price_provider=value("price", "provider", Config.price_provider, str),
        fallback_provider=value("price", "fallback_provider", Config.fallback_provider, str),
        request_timeout_sec=value("price", "request_timeout_sec", Config.request_timeout_sec, float),
        price_cache_hours=value("price", "cache_hours", Config.price_cache_hours, float),
        lookahead_hours=value("price", "lookahead_hours", Config.lookahead_hours, int),
        sample_interval_sec=value("sampling", "interval_sec", Config.sample_interval_sec, float),
        history_samples=value("sampling", "history_samples", Config.history_samples, int),
        busy_core_threshold=value("sampling", "busy_core_threshold", Config.busy_core_threshold, float),
        warm_temp_c=value("thermal", "warm_temp_c", Config.warm_temp_c, float),
        hot_temp_c=value("thermal", "hot_temp_c", Config.hot_temp_c, float),
        critical_temp_c=value("thermal", "critical_temp_c", Config.critical_temp_c, float),
        thermal_recovery_c=value("thermal", "recovery_c", Config.thermal_recovery_c, float),
        promote_samples=value("transitions", "promote_samples", Config.promote_samples, int),
        relax_seconds=value("transitions", "relax_seconds", Config.relax_seconds, float),
        minimum_residence_seconds=value("transitions", "minimum_residence_seconds", Config.minimum_residence_seconds, float),
        database_path=value("storage", "database_path", Config.database_path, str),
        retention_days=value("storage", "retention_days", Config.retention_days, int),
        sample_retention_days=value("storage", "sample_retention_days", Config.sample_retention_days, int),
        dry_run=dry_run or value("general", "dry_run", Config.dry_run, bool),
        manage_cpu=value("cpu", "enabled", Config.manage_cpu, bool),
        manage_nvidia=value("gpu", "nvidia_enabled", Config.manage_nvidia, bool),
        manage_amdgpu=value("gpu", "amdgpu_enabled", Config.manage_amdgpu, bool),
        manage_powercap=value("cpu", "powercap_enabled", Config.manage_powercap, bool),
        external_change_policy=value("general", "external_change_policy", Config.external_change_policy, str).lower(),
        restore_on_exit=value("general", "restore_on_exit", Config.restore_on_exit, bool),
        protected_processes=value("workloads", "protected_processes", (), tuple),
    )

    issues: list[str] = []
    if cfg.area not in {"SE1", "SE2", "SE3", "SE4"}:
        issues.append("[price] area must be SE1, SE2, SE3, or SE4")
    providers = {"elpris_eu", "elprisetjustnu"}
    if cfg.price_provider not in providers:
        issues.append(f"[price] provider is unsupported: {cfg.price_provider}")
    if cfg.fallback_provider not in providers:
        issues.append(f"[price] fallback_provider is unsupported: {cfg.fallback_provider}")
    try:
        ZoneInfo(cfg.timezone)
    except ZoneInfoNotFoundError:
        issues.append(f"[price] timezone is invalid: {cfg.timezone}")
    for label, number in (
        ("request_timeout_sec", cfg.request_timeout_sec),
        ("cache_hours", cfg.price_cache_hours),
        ("interval_sec", cfg.sample_interval_sec),
        ("relax_seconds", cfg.relax_seconds),
        ("minimum_residence_seconds", cfg.minimum_residence_seconds),
    ):
        if number <= 0:
            issues.append(f"{label} must be greater than zero")
    if cfg.lookahead_hours < 1 or cfg.lookahead_hours > 48:
        issues.append("[price] lookahead_hours must be between 1 and 48")
    if cfg.history_samples < 6:
        issues.append("[sampling] history_samples must be at least 6")
    if not 0 <= cfg.busy_core_threshold <= 100:
        issues.append("[sampling] busy_core_threshold must be between 0 and 100")
    if not cfg.warm_temp_c < cfg.hot_temp_c < cfg.critical_temp_c:
        issues.append("thermal thresholds must satisfy warm < hot < critical")
    if not 0 < cfg.thermal_recovery_c < cfg.warm_temp_c:
        issues.append("[thermal] recovery_c must be greater than zero and lower than warm_temp_c")
    if cfg.promote_samples < 1:
        issues.append("[transitions] promote_samples must be at least 1")
    if cfg.retention_days < 1 or cfg.sample_retention_days < 1:
        issues.append("retention periods must be at least one day")
    if cfg.external_change_policy not in {"observe", "manage", "yield"}:
        issues.append("external_change_policy must be observe, manage, or yield")
    if not cfg.database_path.strip():
        issues.append("[storage] database_path cannot be empty")
    if issues:
        raise ConfigError("; ".join(issues))
    return cfg
