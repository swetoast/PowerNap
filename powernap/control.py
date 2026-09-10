from __future__ import annotations

from pathlib import Path
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
    ):
        self.capabilities = capabilities
        self.dry_run = dry_run
        self.manage_cpu = manage_cpu
        self.manage_nvidia = manage_nvidia
        self.manage_amdgpu = manage_amdgpu
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

    def plan(self, profile: Profile) -> list[ControlOperation]:
        operations: list[ControlOperation] = []
        if self.manage_cpu:
            operations.extend(self._cpu_plan(profile))
        if self.manage_nvidia:
            operations.extend(self._nvidia_plan(profile))
        if self.manage_amdgpu:
            operations.extend(self._amd_plan(profile))
        return operations

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
            if result.state == ResultState.APPLIED:
                applied.append(result)
                self._remember(result)
            elif result.state == ResultState.FAILED and operation.required:
                rollback = self._rollback(applied)
                results.extend(rollback)
                break
        return results

    def _remember(self, result: OperationResult) -> None:
        key = result.operation.target if result.operation.adapter == "file" else f"nvml://{result.operation.target}/power_limit_w"
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
            if value is None:
                continue
            if target.startswith("nvml://"):
                uuid = target.removeprefix("nvml://").removesuffix("/power_limit_w")
                current = self.expected_state.get(target)
                operations.append(ControlOperation("nvidia", uuid, current, value, False))
            else:
                operations.append(ControlOperation("file", target, read_text(Path(target)), value, False))
        return self.apply_transaction(list(reversed(operations)))
