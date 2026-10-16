"""Shared GPU worker planning and guarded local CUDA execution."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
from typing import Any, Iterable, Protocol

from ..contracts import GPUExecutionPlanV1, GPUWorkerEventV1, GPUWorkerExperimentV1


@dataclass(frozen=True, slots=True)
class GPUPreflightResult:
    available: bool
    reasons: tuple[str, ...]
    telemetry: dict[str, object]


class LocalContainerTransport(Protocol):
    def start(self, command: tuple[str, ...]) -> str: ...
    def inspect(self, container_id: str) -> dict[str, Any]: ...
    def logs(self, container_id: str) -> Iterable[str]: ...
    def stop(self, container_id: str) -> None: ...
    def remove(self, container_id: str) -> None: ...


class DockerCLITransport:
    """Small Docker CLI boundary with no shell interpolation."""

    def start(self, command: tuple[str, ...]) -> str:
        process = subprocess.run(command, capture_output=True, text=True, check=False)
        if process.returncode != 0:
            raise RuntimeError(f"Docker launch failed: {process.stderr.strip()}")
        container_id = process.stdout.strip()
        if re.fullmatch(r"[0-9a-f]{12,64}", container_id) is None:
            raise RuntimeError("Docker launch did not return a valid container id")
        return container_id

    def inspect(self, container_id: str) -> dict[str, Any]:
        _container_id(container_id)
        process = subprocess.run(
            ["docker", "inspect", container_id], capture_output=True, text=True, check=False,
        )
        if process.returncode != 0:
            raise RuntimeError(f"Docker inspect failed: {process.stderr.strip()}")
        payload = json.loads(process.stdout)
        if not isinstance(payload, list) or len(payload) != 1:
            raise RuntimeError("Docker inspect returned an invalid response")
        state = payload[0].get("State", {})
        return {
            "id": container_id, "running": bool(state.get("Running")),
            "status": str(state.get("Status", "unknown")),
            "exit_code": state.get("ExitCode"), "error": state.get("Error") or None,
        }

    def logs(self, container_id: str) -> tuple[str, ...]:
        _container_id(container_id)
        process = subprocess.run(
            ["docker", "logs", container_id], capture_output=True, text=True, check=False,
        )
        if process.returncode != 0:
            raise RuntimeError(f"Docker logs failed: {process.stderr.strip()}")
        return tuple(process.stdout.splitlines())

    def stop(self, container_id: str) -> None:
        _container_id(container_id)
        process = subprocess.run(
            ["docker", "stop", "--time", "30", container_id],
            capture_output=True, text=True, check=False,
        )
        if process.returncode != 0 and "No such container" not in process.stderr:
            raise RuntimeError(f"Docker stop failed: {process.stderr.strip()}")

    def remove(self, container_id: str) -> None:
        _container_id(container_id)
        process = subprocess.run(
            ["docker", "rm", container_id], capture_output=True, text=True, check=False,
        )
        if process.returncode != 0 and "No such container" not in process.stderr:
            raise RuntimeError(f"Docker remove failed: {process.stderr.strip()}")


class LocalCUDAExecutionProvider:
    provider_id = "local-cuda"

    def __init__(self, transport: LocalContainerTransport | None = None):
        self.transport = transport or DockerCLITransport()

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
    def docker_command(
        plan: GPUExecutionPlanV1, *, data_dir: Path, command: tuple[str, ...],
        container_name: str | None = None,
    ) -> tuple[str, ...]:
        if plan.provider != "local-cuda":
            raise ValueError("plan provider must be local-cuda")
        if not data_dir.is_dir():
            raise ValueError("local CUDA data directory does not exist")
        uid, gid = os.getuid(), os.getgid()
        if uid == 0:
            raise ValueError("local CUDA worker refuses to run as the root host user")
        image = f"{plan.image}@{plan.image_digest}"
        base = (
            "docker", "run", "--detach", "--gpus", str(plan.gpu_count), "--ipc=host",
            "--shm-size=64g", "--read-only", "--cap-drop=ALL",
            "--security-opt=no-new-privileges", "--network=none", "--pids-limit=4096",
            f"--user={uid}:{gid}", "--env=HOME=/tmp", "--tmpfs=/tmp:rw,noexec,nosuid,size=8g",
            "--mount", f"type=bind,src={data_dir.resolve()},dst=/workspace",
        )
        named = base if container_name is None else (*base, "--name", _safe_name(container_name))
        return (*named, image, *command)

    def stage(self, plan: GPUExecutionPlanV1, *, data_dir: Path, experiment: GPUWorkerExperimentV1) -> Path:
        if plan.provider != self.provider_id:
            raise ValueError("execution plan is not for local CUDA")
        root = data_dir.resolve()
        run_root = root / "runs" / _safe_name(experiment.experiment_id)
        expected_output = f"/workspace/runs/{experiment.experiment_id}/output"
        expected_cancel = f"/workspace/runs/{experiment.experiment_id}/cancel.requested"
        if experiment.output_dir != expected_output or experiment.cancellation_file != expected_cancel:
            raise ValueError("local CUDA output and cancellation paths must use the managed run directory")
        updates: dict[str, str] = {}
        for field in (
            "environment_manifest", "model_path", "caller_model_path",
            "rollout_dataset_path", "validation_dataset_path", "verl_checkout", "tau2_checkout",
        ):
            value = getattr(experiment, field)
            if value is None or value.startswith("/opt/audioenv/checkouts/"):
                continue
            updates[field] = _container_path(root, Path(value))
        staged = experiment.model_copy(update=updates)
        path = run_root / "experiment.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(staged.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return path

    def launch(
        self, plan: GPUExecutionPlanV1, *, data_dir: Path, experiment_path: Path,
        run_name: str, confirmation_name: str,
    ) -> dict[str, Any]:
        if confirmation_name != run_name:
            raise ValueError("typed confirmation must exactly match the run name")
        root = data_dir.resolve()
        container_experiment = _container_path(root, experiment_path)
        name = _safe_name(f"audioenv-{experiment_path.parent.name}")
        worker = (
            "timeout", "--signal=TERM", str(int(plan.max_runtime_hours * 3600)),
            "audioenv-gpu-worker", "run", container_experiment,
        )
        command = self.docker_command(
            plan, data_dir=root, command=worker, container_name=name,
        )
        return {"container_id": self.transport.start(command), "command": list(command), "name": name}

    def reconnect(self, container_id: str) -> dict[str, Any]:
        return self.transport.inspect(_container_id(container_id))

    def stream_events(
        self, container_id: str, *, experiment_id: str, after_sequence: int = 0,
    ) -> tuple[GPUWorkerEventV1, ...]:
        events: list[GPUWorkerEventV1] = []
        expected = after_sequence + 1
        for line in self.transport.logs(_container_id(container_id)):
            try:
                event = GPUWorkerEventV1.model_validate_json(line)
            except Exception:
                continue
            if event.sequence <= after_sequence:
                continue
            if event.experiment_id != experiment_id:
                raise ValueError("local CUDA event belongs to another experiment")
            if event.sequence != expected:
                raise ValueError("local CUDA event sequence is not contiguous")
            events.append(event)
            expected += 1
        return tuple(events)

    def collect_artifacts(
        self, *, data_dir: Path, experiment_id: str, destination: Path,
    ) -> tuple[Path, ...]:
        output = data_dir.resolve() / "runs" / _safe_name(experiment_id) / "output"
        manifest_path = output / "worker-artifacts.json"
        manifest_payload = manifest_path.read_bytes()
        manifest = json.loads(manifest_payload)
        if not isinstance(manifest, list):
            raise ValueError("worker artifact manifest must be a list")
        destination.mkdir(parents=True, exist_ok=True)
        collected: list[Path] = []
        for item in manifest:
            relative = PurePosixPath(str(item["path"]))
            if relative.is_absolute() or ".." in relative.parts or relative.name == manifest_path.name:
                raise ValueError("unsafe worker artifact path")
            source = (output / Path(*relative.parts)).resolve()
            if not source.is_relative_to(output.resolve()) or not source.is_file():
                raise ValueError("worker artifact is missing or escapes output directory")
            payload = source.read_bytes()
            if len(payload) != int(item["size_bytes"]) or sha256(payload).hexdigest() != item["sha256"]:
                raise ValueError(f"artifact integrity check failed: {relative}")
            target = (destination / Path(*relative.parts)).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError("artifact path escapes destination")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
            collected.append(target)
        (destination / manifest_path.name).write_bytes(manifest_payload)
        return tuple(collected)

    def cancel(self, container_id: str) -> None:
        self.transport.stop(_container_id(container_id))

    def cleanup(self, container_id: str) -> None:
        self.transport.remove(_container_id(container_id))

    @staticmethod
    def write_plan(plan: GPUExecutionPlanV1, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(plan.model_dump(mode="json"), sort_keys=True, indent=2), encoding="utf-8")
        return destination


def _container_path(root: Path, source: Path) -> str:
    resolved = source.resolve()
    if not resolved.exists():
        raise ValueError(f"local CUDA resource does not exist: {source}")
    if not resolved.is_relative_to(root):
        raise ValueError("local CUDA resources must be contained by the mounted data directory")
    return "/workspace/" + resolved.relative_to(root).as_posix()


def _safe_name(value: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value) is None:
        raise ValueError("unsafe local CUDA identifier")
    return value


def _container_id(value: str) -> str:
    if re.fullmatch(r"[0-9a-f]{12,64}", value) is None:
        raise ValueError("invalid Docker container id")
    return value
