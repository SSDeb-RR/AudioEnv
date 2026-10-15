"""Shared GPU worker planning and local CUDA preflight."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import subprocess

from ..contracts import GPUExecutionPlanV1


@dataclass(frozen=True, slots=True)
class GPUPreflightResult:
    available: bool
    reasons: tuple[str, ...]
    telemetry: dict[str, object]


class LocalCUDAExecutionProvider:
    provider_id = "local-cuda"

    def preflight(
        self, plan: GPUExecutionPlanV1, *, model_path: Path,
        required_free_storage_gb: int = 200,
    ) -> GPUPreflightResult:
        reasons: list[str] = []
        telemetry: dict[str, object] = {"provider": self.provider_id}
        if plan.provider != self.provider_id:
            reasons.append("execution plan is not for local CUDA")
        if not model_path.exists():
            reasons.append("model checkpoint does not exist")
        if shutil.which("docker") is None:
            reasons.append("docker is unavailable")
        if shutil.which("nvidia-smi") is None:
            reasons.append("nvidia-smi is unavailable")
        else:
            process = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=False,
            )
            devices = []
            for line in process.stdout.splitlines():
                name, _, memory = line.rpartition(",")
                if memory.strip().isdigit():
                    devices.append({"name": name.strip(), "memory_mb": int(memory.strip())})
            telemetry["devices"] = devices
            eligible = [item for item in devices if int(item["memory_mb"]) >= plan.memory_gb_per_gpu * 1000 - 1000]
            if len(eligible) < plan.gpu_count:
                reasons.append(f"requires {plan.gpu_count} eligible GPUs; found {len(eligible)}")
        free_gb = shutil.disk_usage(model_path if model_path.exists() else Path.cwd()).free // (1024 ** 3)
        telemetry["free_storage_gb"] = free_gb
        if free_gb < required_free_storage_gb:
            reasons.append(f"requires {required_free_storage_gb} GB free storage; found {free_gb} GB")
        return GPUPreflightResult(not reasons, tuple(reasons), telemetry)

    @staticmethod
    def docker_command(plan: GPUExecutionPlanV1, *, data_dir: Path, command: tuple[str, ...]) -> tuple[str, ...]:
        if plan.provider != "local-cuda":
            raise ValueError("plan provider must be local-cuda")
        return (
            "docker", "run", "--rm", "--gpus", str(plan.gpu_count), "--ipc=host",
            "--shm-size=64g", "--read-only", "--cap-drop=ALL",
            "--security-opt=no-new-privileges", "--mount",
            f"type=bind,src={data_dir.resolve()},dst=/workspace", plan.image, *command,
        )

    @staticmethod
    def write_plan(plan: GPUExecutionPlanV1, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(plan.model_dump(mode="json"), sort_keys=True, indent=2), encoding="utf-8")
        return destination
