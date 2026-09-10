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
        self.repository.record_capabilities(self.capabilities)
        self.collector = Collector(cfg, self.capabilities)
        self.engine = DecisionEngine(cfg)
        self.transition = TransitionManager.from_state(cfg, self.repository.get_meta("transition_state"))
        self.simulated_transition = TransitionManager.from_state(cfg, self.repository.get_meta("simulated_transition_state"))
        self.controller = Controller(
            self.capabilities,
            cfg.dry_run,
            cfg.manage_cpu,
            cfg.manage_nvidia,
            cfg.manage_amdgpu,
            cfg.manage_powercap,
        )
        self.price = PriceService(
            cfg.area,
            cfg.timezone,
            cfg.request_timeout_sec,
            cfg.price_provider,
            cfg.fallback_provider,
            cfg.price_cache_hours,
            repository=self.repository,
        )
        self.baseline = dict(self.controller.expected_state)
        self.needs_reconcile = True
        self.degraded_reason: str | None = None
        self.last_cycle_monotonic: float | None = None

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
            self.controller.yield_targets(changes)
            logging.warning("External power setting change detected; yielded %d affected controls", len(changes))
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
                if self.last_cycle_monotonic is not None and cycle_start - self.last_cycle_monotonic > self.cfg.sample_interval_sec * 3:
                    self._after_resume()
                self.last_cycle_monotonic = cycle_start
                price = self.price.context(lookahead_hours=self.cfg.lookahead_hours)
                state = self.collector.collect(price)
                decision = self._protect_workload(self.engine.decide(state))
                transition = self.simulated_transition if self.cfg.dry_run else self.transition
                gate = transition.evaluate(decision)
                results = []
                reapply = self._resolve_external_changes()
                transaction = self.controller.transaction_summary(results)
                if gate.allowed or self.needs_reconcile or reapply:
                    if decision.thermal_state.value == "critical":
                        target_profile = Profile.ECO
                    elif self.degraded_reason:
                        target_profile = min(Profile.BALANCED, decision.safety_ceiling)
                    else:
                        target_profile = gate.profile
                    results = self.controller.apply_transaction(self.controller.plan(target_profile))
                    transaction = self.controller.transaction_summary(results)
                    required_failures = transaction["failed_required"] > 0
                    self.needs_reconcile = required_failures
                    if not required_failures:
                        transition.commit(target_profile)
                        if not self.cfg.dry_run:
                            actual = self.controller.snapshot()
                            conflicts = self.controller.external_changes(actual)
                            if conflicts:
                                self.needs_reconcile = True
                                self.degraded_reason = "post-transaction reconciliation failed"
                            elif self.degraded_reason in {"required control verification failed", "post-transaction reconciliation failed"}:
                                self.degraded_reason = None
                    else:
                        self.degraded_reason = "required control verification failed"
                try:
                    self.repository.record_cycle(state, decision, results)
                    self.repository.set_meta("transition_state", self.transition.export_state())
                    self.repository.set_meta("simulated_transition_state", self.simulated_transition.export_state())
                    self.repository.set_meta("last_transaction", transaction)
                    if self.degraded_reason == "database unavailable":
                        self.degraded_reason = None
                except RuntimeError as exc:
                    self.degraded_reason = "database unavailable"
                    logging.error("%s", exc)
                cycles += 1
                if time.time() - last_prune >= 86_400:
                    try:
                        self.repository.prune(self.cfg.sample_retention_days, self.cfg.retention_days)
                        last_prune = time.time()
                    except Exception as exc:
                        self.degraded_reason = "database unavailable"
                        logging.error("Unable to prune PowerNap database: %s", exc)
                status = f"Degraded: {self.degraded_reason}" if self.degraded_reason else f"{gate.profile.name.title()}: {decision.reason}"
                sd_notify(f"WATCHDOG=1\nSTATUS={status}")
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


    def _after_resume(self) -> None:
        logging.info("Resume or long scheduling gap detected; refreshing capabilities")
        refreshed = discover()
        try:
            self.repository.record_capabilities(refreshed)
        except Exception as exc:
            self.degraded_reason = "database unavailable"
            logging.error("Unable to record refreshed capabilities: %s", exc)
        self.capabilities = refreshed
        self.collector.capabilities = refreshed
        yielded = set(self.controller.yielded_targets)
        self.controller = Controller(refreshed, self.cfg.dry_run, self.cfg.manage_cpu, self.cfg.manage_nvidia, self.cfg.manage_amdgpu, self.cfg.manage_powercap)
        self.controller.yielded_targets = yielded
        for target, value in self.controller.expected_state.items():
            self.baseline.setdefault(target, value)
        self.transition = TransitionManager(self.cfg, self.transition.current)
        simulated = getattr(self, "simulated_transition", None)
        self.simulated_transition = TransitionManager(
            self.cfg, simulated.current if simulated is not None else self.transition.current
        )
        history = getattr(self.collector, "history", None)
        if history is not None:
            history.clear()
        self.needs_reconcile = True

    def _interruptible_sleep(self, duration: float) -> None:
        deadline = time.monotonic() + duration
        while not self.stop_requested:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(remaining, 0.5))
