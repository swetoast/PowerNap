from __future__ import annotations

from pathlib import Path
import time
from typing import Mapping

from .capabilities import Capabilities, CPUFreqPolicy, read_text
from .model import ControlOperation, OperationResult, Profile, ResultState

GOVERNORS = {
    Profile.ECO: ("powersave", "conservative", "schedutil", "ondemand", "performance"),
    Profile.BALANCED: ("schedutil", "ondemand", "conservative", "powersave", "performance"),
    Profile.RESPONSIVE: ("schedutil", "ondemand", "performance", "conservative", "powersave"),
    Profile.MAXIMUM: ("performance", "schedutil", "ondemand", "conservative", "powersave"),
}
FREQUENCY_FRACTION = {Profile.ECO: 0.50, Profile.BALANCED: 0.75, Profile.RESPONSIVE: 0.90, Profile.MAXIMUM: 1.0}
GPU_POWER_FRACTION = {Profile.ECO: 0.0, Profile.BALANCED: 0.25, Profile.RESPONSIVE: 0.60, Profile.MAXIMUM: 1.0}
POWERCAP_FRACTION = {Profile.ECO: 0.45, Profile.BALANCED: 0.70, Profile.RESPONSIVE: 0.90, Profile.MAXIMUM: 1.0}
EPP = {
    Profile.ECO: ("power", "balance_power"),
    Profile.BALANCED: ("balance_power", "balance_performance"),
    Profile.RESPONSIVE: ("balance_performance", "performance"),
    Profile.MAXIMUM: ("performance",),
}


def preferred_governor(policy: CPUFreqPolicy, profile: Profile) -> str | None:
    return next((item for item in GOVERNORS[profile] if item in policy.governors), None)


class Controller:
    def __init__(
        self,
        capabilities: Capabilities,
        dry_run: bool,
        manage_cpu: bool = True,
        manage_nvidia: bool = True,
        manage_amdgpu: bool = True,
        manage_powercap: bool = False,
    ):
        self.capabilities = capabilities
        self.dry_run = dry_run
        self.manage_cpu = manage_cpu
        self.manage_nvidia = manage_nvidia
        self.manage_amdgpu = manage_amdgpu
        self.manage_powercap = manage_powercap
        self.yielded_targets: set[str] = set()
        self.retry_not_before: dict[str, float] = {}
        self.failure_counts: dict[str, int] = {}
        self.expected_state = self.snapshot()

    def snapshot(self) -> dict[str, str | float | None]:
        state: dict[str, str | float | None] = {}
        if self.manage_cpu:
            for policy in self.capabilities.cpu_policies:
                base = Path(policy.path)
                for filename in ("scaling_governor", "scaling_min_freq", "scaling_max_freq", "energy_performance_preference"):
                    path = str(base / filename)
                    value = read_text(Path(path))
                    if value is not None:
                        state[path] = value
        if self.manage_amdgpu:
            for gpu in self.capabilities.amd_gpus:
                path = str(Path(gpu.path) / "power_dpm_force_performance_level")
                value = read_text(Path(path))
                if value is not None:
                    state[path] = value
        if self.manage_powercap:
            for zone in self.capabilities.powercap_zones:
                for constraint in zone.constraints:
                    value = read_text(Path(constraint.path))
                    if value is not None:
                        state[constraint.path] = value
        if self.manage_nvidia:
            state.update(self._nvidia_snapshot())
        return state

    def _nvidia_snapshot(self) -> dict[str, float]:
        try:
            import pynvml
            pynvml.nvmlInit()
        except Exception:
            return {}
        state = {}
        try:
            for gpu in self.capabilities.nvidia_gpus:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByUUID(gpu.uuid)
                    state[f"nvml://{gpu.uuid}/power_limit_w"] = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
                except Exception:
                    continue
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
        return state

    def external_changes(self, current: Mapping[str, object] | None = None) -> dict[str, tuple[object, object]]:
        current_state = dict(current) if current is not None else self.snapshot()
        return {
            key: (expected, current_state.get(key))
            for key, expected in self.expected_state.items()
            if key in current_state and current_state[key] != expected
        }

    @staticmethod
    def operation_key(operation: ControlOperation) -> str:
        return operation.target if operation.adapter == "file" else f"nvml://{operation.target}/power_limit_w"

    def yield_targets(self, targets) -> None:
        self.yielded_targets.update(targets)

    def plan(self, profile: Profile) -> list[ControlOperation]:
        operations: list[ControlOperation] = []
        if self.manage_cpu:
            operations.extend(self._cpu_plan(profile))
        if self.manage_powercap:
            operations.extend(self._powercap_plan(profile))
        if self.manage_nvidia:
            operations.extend(self._nvidia_plan(profile))
        if self.manage_amdgpu:
            operations.extend(self._amd_plan(profile))
        now = time.monotonic()
        filtered = [
            operation for operation in operations
            if self.operation_key(operation) not in self.yielded_targets
            and self.retry_not_before.get(self.operation_key(operation), 0.0) <= now
        ]
        return self._order_operations(filtered, profile)

    @staticmethod
    def _order_operations(operations: list[ControlOperation], profile: Profile) -> list[ControlOperation]:
        def numeric(value):
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        limit_names = {"scaling_max_freq", "power_limit_w"}
        limit_changes = []
        for item in operations:
            name = Path(item.target).name if item.adapter == "file" else "power_limit_w"
            if name in limit_names or name.endswith("_power_limit_uw"):
                old, new = numeric(item.old_value), numeric(item.requested_value)
                if old is not None and new is not None and new != old:
                    limit_changes.append(new > old)
        increasing = any(limit_changes) if limit_changes else profile > Profile.BALANCED

        def priority(item):
            name = Path(item.target).name if item.adapter == "file" else "power_limit_w"
            limit = name in limit_names or name.endswith("_power_limit_uw")
            policy = name in {"scaling_governor", "energy_performance_preference", "power_dpm_force_performance_level"}
            if increasing:
                group = 0 if limit else 1 if policy else 2
            else:
                group = 0 if policy else 1 if limit else 2
            return group, item.target
        return sorted(operations, key=priority)

    def _cpu_plan(self, profile: Profile) -> list[ControlOperation]:
        operations = []
        for policy in self.capabilities.cpu_policies:
            base = Path(policy.path)
            governor_path = base / "scaling_governor"
            max_path = base / "scaling_max_freq"
            epp_path = base / "energy_performance_preference"
            governor = preferred_governor(policy, profile)
            current_governor = read_text(governor_path)
            current_max = read_text(max_path)
            current_epp = read_text(epp_path)
            target_max = None
            if policy.hw_min_khz is not None and policy.hw_max_khz is not None:
                target_max = round(policy.hw_min_khz + (policy.hw_max_khz - policy.hw_min_khz) * FREQUENCY_FRACTION[profile])
                target_max = max(policy.hw_min_khz, min(policy.hw_max_khz, target_max))
            target_epp = next((item for item in EPP[profile] if item in policy.epp_available), None)

            try:
                current_max_value = int(current_max) if current_max is not None else None
            except ValueError:
                current_max_value = None
            increasing = target_max is not None and current_max_value is not None and target_max > current_max_value
            if increasing and current_max != str(target_max):
                operations.append(ControlOperation("file", str(max_path), current_max, target_max))
            if governor and current_governor != governor:
                operations.append(ControlOperation("file", str(governor_path), current_governor, governor))
            if target_epp and current_epp != target_epp:
                operations.append(ControlOperation("file", str(epp_path), current_epp, target_epp, False))
            if not increasing and target_max is not None and current_max != str(target_max):
                operations.append(ControlOperation("file", str(max_path), current_max, target_max))
        return operations

    def _powercap_plan(self, profile: Profile) -> list[ControlOperation]:
        operations = []
        fraction = POWERCAP_FRACTION[profile]
        for zone in self.capabilities.powercap_zones:
            if zone.enabled is False:
                continue
            eligible = [
                constraint for constraint in zone.constraints
                if constraint.name.lower() in {"long_term", "long term", "slow", "package"}
            ]
            for constraint in eligible[:1]:
                if constraint.power_limit_uw is None:
                    continue
                minimum = constraint.min_power_uw
                maximum = constraint.max_power_uw
                if minimum is None or maximum is None or maximum < minimum:
                    continue
                target = int(round(minimum + (maximum - minimum) * fraction))
                target = max(minimum, min(maximum, target))
                operations.append(ControlOperation(
                    "file", constraint.path, read_text(Path(constraint.path)), str(target), False
                ))
        return operations

    def _nvidia_plan(self, profile: Profile) -> list[ControlOperation]:
        operations = []
        for gpu in self.capabilities.nvidia_gpus:
            if gpu.min_power_w is None or gpu.max_power_w is None:
                continue
            target = round(gpu.min_power_w + (gpu.max_power_w - gpu.min_power_w) * GPU_POWER_FRACTION[profile], 3)
            target = max(gpu.min_power_w, min(gpu.max_power_w, target))
            current = self.expected_state.get(f"nvml://{gpu.uuid}/power_limit_w", gpu.current_power_w)
            if current is None or abs(float(current) - target) >= 0.01:
                operations.append(ControlOperation("nvidia", gpu.uuid, current, target))
        return operations

    def _amd_plan(self, profile: Profile) -> list[ControlOperation]:
        operations = []
        level = {Profile.ECO: "low", Profile.BALANCED: "auto", Profile.RESPONSIVE: "auto", Profile.MAXIMUM: "high"}[profile]
        for gpu in self.capabilities.amd_gpus:
            path = Path(gpu.path) / "power_dpm_force_performance_level"
            current = read_text(path)
            if current is not None and current != level:
                operations.append(ControlOperation("file", str(path), current, level, False))
        return operations

    def apply_transaction(self, operations: list[ControlOperation]) -> list[OperationResult]:
        results: list[OperationResult] = []
        applied: list[OperationResult] = []
        for operation in operations:
            result = self._apply(operation)
            results.append(result)
            key = self.operation_key(operation)
            if result.state == ResultState.APPLIED:
                applied.append(result)
                self._remember(result)
                self.failure_counts.pop(key, None)
                self.retry_not_before.pop(key, None)
            elif result.state == ResultState.FAILED:
                failures = self.failure_counts.get(key, 0) + 1
                self.failure_counts[key] = failures
                self.retry_not_before[key] = time.monotonic() + min(300.0, 5.0 * (2 ** (failures - 1)))
            if result.state == ResultState.FAILED and operation.required:
                rollback = self._rollback(applied)
                results.extend(rollback)
                break
        return results

    def infer_applied_profile(self) -> Profile | None:
        if not self.expected_state:
            return None
        yielded = self.yielded_targets
        retry = self.retry_not_before
        self.yielded_targets = set()
        self.retry_not_before = {}
        try:
            matches = [profile for profile in Profile if not self.plan(profile)]
        finally:
            self.yielded_targets = yielded
            self.retry_not_before = retry
        return matches[0] if len(matches) == 1 else None

    @staticmethod
    def transaction_summary(results: list[OperationResult]) -> dict:
        counts = {state.value: 0 for state in ResultState}
        for result in results:
            counts[result.state.value] += 1
        failed_required = sum(
            1 for result in results
            if result.operation.required and result.state == ResultState.FAILED
        )
        failed_optional = sum(
            1 for result in results
            if not result.operation.required and result.state == ResultState.FAILED
        )
        rollback_results = [
            result for result in results
            if not result.operation.required and result.operation.old_value == result.operation.requested_value
        ]
        if failed_required:
            state = "failed"
        elif failed_optional:
            state = "partially_applied"
        elif counts[ResultState.SIMULATED.value]:
            state = "simulated"
        elif counts[ResultState.UNSUPPORTED.value]:
            state = "partially_applied"
        else:
            state = "applied"
        return {
            "state": state,
            "operations": len(results),
            "counts": counts,
            "failed_required": failed_required,
            "failed_optional": failed_optional,
            "rollback_failures": sum(item.state == ResultState.FAILED for item in rollback_results),
        }

    def _remember(self, result: OperationResult) -> None:
        key = self.operation_key(result.operation)
        self.expected_state[key] = result.verified_value

    def _apply(self, operation: ControlOperation) -> OperationResult:
        if self.dry_run:
            return OperationResult(operation, operation.old_value, ResultState.SIMULATED)
        if operation.adapter == "file":
            return self._write_file(operation)
        if operation.adapter == "nvidia":
            return self._write_nvidia(operation)
        return OperationResult(operation, None, ResultState.UNSUPPORTED, "unknown control adapter")

    def _write_file(self, operation: ControlOperation) -> OperationResult:
        path = Path(operation.target)
        try:
            path.write_text(str(operation.requested_value), encoding="utf-8")
            verified = read_text(path)
        except OSError as exc:
            return OperationResult(operation, read_text(path), ResultState.FAILED, str(exc))
        if verified != str(operation.requested_value):
            return OperationResult(operation, verified, ResultState.FAILED, "read-back did not match requested value")
        return OperationResult(operation, verified, ResultState.APPLIED)

    def _write_nvidia(self, operation: ControlOperation) -> OperationResult:
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByUUID(operation.target)
            pynvml.nvmlDeviceSetPowerManagementLimit(handle, round(float(operation.requested_value) * 1000))
            verified = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
            if abs(verified - float(operation.requested_value)) >= 0.01:
                return OperationResult(operation, verified, ResultState.FAILED, "read-back did not match requested value")
            return OperationResult(operation, verified, ResultState.APPLIED)
        except Exception as exc:
            return OperationResult(operation, None, ResultState.FAILED, str(exc))
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    def _rollback(self, applied: list[OperationResult]) -> list[OperationResult]:
        results = []
        for item in reversed(applied):
            old = item.operation.old_value
            if old is None:
                continue
            rollback_op = ControlOperation(item.operation.adapter, item.operation.target, item.verified_value, old, False)
            rollback_result = self._apply(rollback_op)
            results.append(rollback_result)
            if rollback_result.state == ResultState.APPLIED:
                self._remember(rollback_result)
        return results

    def restore(self, snapshot: Mapping[str, str | float | None]) -> list[OperationResult]:
        operations = []
        for target, value in snapshot.items():
            if target in self.yielded_targets or value is None:
                continue
            if target.startswith("nvml://"):
                uuid = target.removeprefix("nvml://").removesuffix("/power_limit_w")
                current = self.expected_state.get(target)
                operations.append(ControlOperation("nvidia", uuid, current, value, False))
            else:
                operations.append(ControlOperation("file", target, read_text(Path(target)), value, False))
        return self.apply_transaction(list(reversed(operations)))
