from __future__ import annotations

import logging
import signal
import time
from dataclasses import replace
from pathlib import Path

from .capabilities import discover
from .config import Config
from .control import Controller
from .database import Repository
from .engine import DecisionEngine, TransitionManager
from .model import Profile
from .notify import sd_notify
from .price import PriceService
from .runtime import Collector
from .workloads import protected_active


class Daemon:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.stop_requested = False
        self.capabilities = discover()
        self.repository = Repository(Path(cfg.database_path))
        self.collector = Collector(cfg, self.capabilities)
        self.engine = DecisionEngine(cfg)
        self.transition = TransitionManager(cfg)
        self.controller = Controller(
            self.capabilities,
            cfg.dry_run,
            cfg.manage_cpu,
            cfg.manage_nvidia,
            cfg.manage_amdgpu,
        )
        self.price = PriceService(
            cfg.area,
            cfg.timezone,
            cfg.request_timeout_sec,
            cfg.price_provider,
            cfg.fallback_provider,
            cfg.price_cache_hours,
        )
        self.baseline = dict(self.controller.expected_state)
        self.yielded = False
        self.needs_reconcile = True

    def request_stop(self, *_args) -> None:
        self.stop_requested = True

    def _resolve_external_changes(self) -> bool:
        changes = self.controller.external_changes()
        if not changes:
            return False
        if self.cfg.external_change_policy == "observe":
            self.controller.expected_state = self.controller.snapshot()
            logging.info("Accepted %d externally changed power settings", len(changes))
            return False
        elif self.cfg.external_change_policy == "yield":
            self.yielded = True
            logging.warning("External power setting change detected; PowerNap yielded control")
            return False
        logging.warning("External power setting change detected; selected profile will be reapplied")
        return True

    def _protect_workload(self, decision):
        active, names = protected_active(self.cfg.protected_processes)
        if not active or decision.recommended >= Profile.RESPONSIVE:
            return decision
        recommended = min(Profile.RESPONSIVE, decision.safety_ceiling)
        return replace(
            decision,
            demand_floor=Profile.RESPONSIVE,
            recommended=recommended,
            reason="Responsive required by protected workload: " + ", ".join(names) + ".",
        )

    def run(self, max_cycles: int | None = None) -> int:
        signal.signal(signal.SIGTERM, self.request_stop)
        signal.signal(signal.SIGINT, self.request_stop)
        sd_notify("READY=1\nSTATUS=PowerNap running")
        cycles = 0
        last_prune = 0.0
        try:
            while not self.stop_requested and (max_cycles is None or cycles < max_cycles):
                cycle_start = time.monotonic()
                price = self.price.context(lookahead_hours=self.cfg.lookahead_hours)
                state = self.collector.collect(price)
                decision = self._protect_workload(self.engine.decide(state))
                gate = self.transition.evaluate(decision)
                results = []
                reapply = self._resolve_external_changes()
                if (gate.allowed or self.needs_reconcile or reapply) and not self.yielded:
                    results = self.controller.apply_transaction(self.controller.plan(gate.profile))
                    self.needs_reconcile = any(item.operation.required and item.state.value == "failed" for item in results)
                self.repository.record_cycle(state, decision, results)
                cycles += 1
                if time.time() - last_prune >= 86_400:
                    self.repository.prune(self.cfg.sample_retention_days, self.cfg.retention_days)
                    last_prune = time.time()
                sd_notify(f"WATCHDOG=1\nSTATUS={gate.profile.name.title()}: {decision.reason}")
                remaining = self.cfg.sample_interval_sec - (time.monotonic() - cycle_start)
                if remaining > 0:
                    self._interruptible_sleep(remaining)
            return 0
        finally:
            if self.cfg.restore_on_exit and not self.cfg.dry_run:
                restore_results = self.controller.restore(self.baseline)
                failed = [item for item in restore_results if item.state.value == "failed"]
                if failed:
                    logging.error("Failed to restore %d baseline power settings", len(failed))
            self.repository.close()
            sd_notify("STOPPING=1\nSTATUS=PowerNap stopping")

    def _interruptible_sleep(self, duration: float) -> None:
        deadline = time.monotonic() + duration
        while not self.stop_requested:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(remaining, 0.5))
