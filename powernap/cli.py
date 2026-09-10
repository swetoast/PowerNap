from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

from . import __version__
from .capabilities import discover
from .config import ConfigError, load_config
from .control import Controller
from .daemon import Daemon
from .database import Repository
from .engine import DecisionEngine
from .price import PriceService
from .runtime import Collector


def _config_path(value: Path | None) -> Path | None:
    if value is not None:
        return value
    system_path = Path("/etc/powernap/powernap.conf")
    return system_path if system_path.exists() else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="powernap")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "capabilities", "check", "status", "report", "prices"):
        subparsers.add_parser(command)
    once = subparsers.add_parser("once")
    once.add_argument("--no-price", action="store_true")
    args = parser.parse_args(argv)
    try:
        cfg = load_config(_config_path(args.config), args.dry_run)
    except ConfigError as exc:
        parser.error(str(exc))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.command == "run":
        try:
            return Daemon(cfg).run()
        except RuntimeError as exc:
            logging.error("%s", exc)
            return 3
    capabilities = discover()
    if args.command == "capabilities":
        print(json.dumps(capabilities.to_dict(), indent=2))
        return 0
    if args.command == "check":
        errors, warnings = [], []
        if cfg.manage_cpu and not capabilities.cpu_policies:
            errors.append("CPU control is enabled but no CPUFreq policies were found")
        if "intel_pstate=disable" in capabilities.kernel_cmdline:
            warnings.append("intel_pstate is disabled by the kernel command line")
        if cfg.manage_nvidia and not capabilities.nvidia_gpus:
            warnings.append("NVIDIA management is enabled but no NVML-manageable GPU was found")
        if cfg.manage_amdgpu and not capabilities.amd_gpus:
            warnings.append("AMDGPU management is enabled but no AMDGPU device was found")
        if cfg.manage_powercap and not any(zone.constraints for zone in capabilities.powercap_zones):
            errors.append("Power-cap management is enabled but no bounded RAPL constraints were found")
        elif cfg.manage_powercap and not any(
            constraint.min_power_uw is not None and constraint.max_power_uw is not None
            for zone in capabilities.powercap_zones for constraint in zone.constraints
        ):
            errors.append("Power-cap management is enabled but discovered constraints have no safe bounds")
        print(json.dumps({
            "version": __version__,
            "valid": not errors,
            "dry_run": cfg.dry_run,
            "errors": errors,
            "warnings": warnings,
            "capabilities": capabilities.to_dict(),
        }, indent=2))
        return 1 if errors else 0
    if args.command in {"status", "report"}:
        repository = Repository(Path(cfg.database_path))
        try:
            data = repository.report(1 if args.command == "status" else 50)
        finally:
            repository.close()
        print(json.dumps(data, indent=2))
        return 0

    repository = Repository(Path(cfg.database_path))
    try:
        repository.record_capabilities(capabilities)
        service = PriceService(
            cfg.area,
            cfg.timezone,
            cfg.request_timeout_sec,
            cfg.price_provider,
            cfg.fallback_provider,
            cfg.price_cache_hours,
            repository=repository,
        )
        if args.command == "prices":
            print(json.dumps(asdict(service.context(lookahead_hours=cfg.lookahead_hours)), indent=2))
            return 0
        price = None if args.no_price else service.context(lookahead_hours=cfg.lookahead_hours)
        state = Collector(cfg, capabilities).collect(price)
        decision = DecisionEngine(cfg).decide(state)
        controller = Controller(capabilities, cfg.dry_run, cfg.manage_cpu, cfg.manage_nvidia, cfg.manage_amdgpu, cfg.manage_powercap)
        plan = controller.plan(decision.recommended)
        results = controller.apply_transaction(plan)
        print(json.dumps({
            "state": asdict(state),
            "decision": decision.to_dict(),
            "plan": [asdict(item) for item in plan],
            "results": [asdict(item) for item in results],
        }, indent=2, default=str))
        return 0 if all(item.state.value in {"applied", "simulated"} for item in results) else 2
    finally:
        if "service" in locals():
            service.close()
        repository.close()


if __name__ == "__main__":
    raise SystemExit(main())
