"""Guarded RunPod Secure Cloud provider contract.

No network call occurs at import, application startup, or preflight. A caller
must inject a v2 API transport, request a quote, satisfy budget/runtime limits,
and repeat the exact run name before provisioning can occur.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
from typing import Any, Protocol
from urllib.request import Request, urlopen

from ..contracts import GPUWorkerEventV1, GPUWorkerExperimentV1


class RunPodTransport(Protocol):
    def list_gpu_types(self) -> list[dict[str, Any]]: ...
    def create_pod(self, payload: dict[str, Any]) -> dict[str, Any]: ...
    def get_pod(self, pod_id: str) -> dict[str, Any]: ...
    def stop_pod(self, pod_id: str) -> dict[str, Any]: ...


class RunPodRemoteExecutor(Protocol):
    def mkdir(self, path: str) -> None: ...
    def upload_bytes(self, path: str, payload: bytes) -> None: ...
    def start(self, command: tuple[str, ...], *, stdout_path: str, exit_path: str, pid_path: str) -> None: ...
    def read_bytes(self, path: str) -> bytes: ...
    def touch(self, path: str) -> None: ...
    def download(self, remote_path: str, local_path: Path) -> None: ...


@dataclass(frozen=True, slots=True)
class OpenSSHRemoteExecutor:
    """Strict-host-key SSH/SCP transport for a provisioned Pod.

    Credentials stay in local files referenced by path. They are never added to
    an experiment bundle, command payload, or persisted provider record.
    """

    host: str
    key_path: Path
    known_hosts_path: Path
    port: int = 22
    user: str = "root"

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.:-]*", self.host):
            raise ValueError("a valid RunPod SSH host is required")
        if not re.fullmatch(r"[a-z_][a-z0-9_-]{0,31}", self.user):
            raise ValueError("a valid SSH user is required")
        if not 1 <= self.port <= 65535:
            raise ValueError("SSH port must be between 1 and 65535")
        if not self.key_path.is_file():
            raise ValueError("RunPod SSH private key does not exist")
        if not self.known_hosts_path.is_file():
            raise ValueError("RunPod known_hosts file does not exist")

    @property
    def _ssh_prefix(self) -> tuple[str, ...]:
        return (
            "ssh", "-i", str(self.key_path), "-p", str(self.port),
            "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
            "-o", f"UserKnownHostsFile={self.known_hosts_path}",
            f"{self.user}@{self.host}",
        )

    @property
    def _scp_prefix(self) -> tuple[str, ...]:
        return (
            "scp", "-i", str(self.key_path), "-P", str(self.port),
            "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
            "-o", f"UserKnownHostsFile={self.known_hosts_path}",
        )

    def _run_ssh(self, command: tuple[str, ...], *, allow_missing: bool = False) -> bytes:
        result = subprocess.run(
            (*self._ssh_prefix, shlex.join(command)), capture_output=True, check=False,
        )
        if result.returncode:
            if allow_missing and result.returncode == 1:
                raise FileNotFoundError(command[-1])
            message = result.stderr.decode(errors="replace").strip()
            raise RuntimeError(f"RunPod SSH command failed ({result.returncode}): {message}")
        return result.stdout

    def mkdir(self, path: str) -> None:
        self._run_ssh(("mkdir", "-p", "--", _managed_remote_path(path)))

    def upload_bytes(self, path: str, payload: bytes) -> None:
        remote_path = _managed_remote_path(path)
        with tempfile.NamedTemporaryFile() as temporary:
            temporary.write(payload)
            temporary.flush()
            result = subprocess.run(
                (*self._scp_prefix, temporary.name, f"{self.user}@{self.host}:{remote_path}"),
                capture_output=True, check=False,
            )
        if result.returncode:
            message = result.stderr.decode(errors="replace").strip()
            raise RuntimeError(f"RunPod SCP upload failed ({result.returncode}): {message}")

    def start(
        self, command: tuple[str, ...], *, stdout_path: str, exit_path: str, pid_path: str,
    ) -> None:
        if not command:
            raise ValueError("remote worker command cannot be empty")
        stdout_path = _managed_remote_path(stdout_path)
        exit_path = _managed_remote_path(exit_path)
        pid_path = _managed_remote_path(pid_path)
        worker = shlex.join(command)
        body = f"{worker}; code=$?; printf '%s\\n' \"$code\" > {shlex.quote(exit_path)}"
        script = (
            f"nohup sh -c {shlex.quote(body)} > {shlex.quote(stdout_path)} 2>&1 </dev/null & "
            f"printf '%s\\n' \"$!\" > {shlex.quote(pid_path)}"
        )
        self._run_ssh(("sh", "-c", script))

    def read_bytes(self, path: str) -> bytes:
        return self._run_ssh(("cat", "--", _managed_remote_path(path)), allow_missing=True)

    def touch(self, path: str) -> None:
        self._run_ssh(("touch", "--", _managed_remote_path(path)))

    def download(self, remote_path: str, local_path: Path) -> None:
        remote_path = _managed_remote_path(remote_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            (*self._scp_prefix, f"{self.user}@{self.host}:{remote_path}", str(local_path)),
            capture_output=True, check=False,
        )
        if result.returncode:
            message = result.stderr.decode(errors="replace").strip()
            raise RuntimeError(f"RunPod SCP download failed ({result.returncode}): {message}")


class RunPodRESTV2Transport:
    """Credential-lazy REST transport; secret values are never serialized."""

    def __init__(self, *, api_key_env: str = "RUNPOD_API_KEY", base_url: str = "https://rest.runpod.io"):
        self.api_key_env, self.base_url = api_key_env, base_url.rstrip("/")

    def list_gpu_types(self) -> list[dict[str, Any]]:
        payload = self._request("GET", "/v2/catalog/gpus")
        return list(payload if isinstance(payload, list) else payload.get("gpus", []))

    def create_pod(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/v2/pods", payload)

    def get_pod(self, pod_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v2/pods/{pod_id}")

    def stop_pod(self, pod_id: str) -> dict[str, Any]:
        return self._request("POST", f"/v2/pods/{pod_id}/action", {"action": "stop"})

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None):
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"missing secret environment variable: {self.api_key_env}")
        data = json.dumps(payload).encode() if payload is not None else None
        request = Request(
            self.base_url + path, data=data, method=method,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )
        with urlopen(request, timeout=30) as response:  # noqa: S310 - fixed configured API boundary
            return json.loads(response.read())


@dataclass(frozen=True, slots=True)
class RunPodConfig:
    run_name: str
    image: str = "audioenv/qwen3-omni:v0.5.0"
    image_digest: str = ""
    gpu_count: int = 8
    gpu_memory_gb: int = 80
    max_runtime_hours: float = 8.0
    max_total_budget_usd: float = 200.0
    network_volume_id: str | None = None
    ssh_key_secret_name: str = "RUNPOD_SSH_KEY"
    api_key_secret_name: str = "RUNPOD_API_KEY"

    def __post_init__(self):
        if not self.image_digest.startswith("sha256:") or len(self.image_digest) != 71:
            raise ValueError("RunPod image must include an immutable sha256 digest")
        if any(character not in "0123456789abcdef" for character in self.image_digest[7:]):
            raise ValueError("RunPod image digest must be lowercase hexadecimal")
        if self.max_runtime_hours <= 0 or self.max_total_budget_usd < 0:
            raise ValueError("RunPod runtime and budget limits are required")


@dataclass(frozen=True, slots=True)
class RunPodQuote:
    gpu_type_id: str
    gpu_count: int
    hourly_price_usd: float
    maximum_cost_usd: float
    available: bool


class RunPodSecureCloudProvider:
    provider_id = "runpod-secure-cloud"

    def __init__(self, transport: RunPodTransport):
        self.transport = transport

    def quote(self, config: RunPodConfig) -> RunPodQuote:
        if config.gpu_count != 8 or config.gpu_memory_gb < 80:
            raise ValueError("V1 GPU gate requires exactly 8 GPUs with at least 80 GB each")
        eligible = [
            item for item in self.transport.list_gpu_types()
            if int(item.get("memory_gb", item.get("memoryInGb", 0))) >= config.gpu_memory_gb
            and int(item.get("secure_cloud_available", item.get("secureCloudAvailable", 0))) >= config.gpu_count
        ]
        if not eligible:
            return RunPodQuote("unavailable", config.gpu_count, 0.0, 0.0, False)
        selected = min(eligible, key=_gpu_price)
        hourly = _gpu_price(selected) * config.gpu_count
        maximum = hourly * config.max_runtime_hours
        return RunPodQuote(str(selected["id"]), config.gpu_count, hourly, maximum, maximum <= config.max_total_budget_usd)

    def provision(self, config: RunPodConfig, quote: RunPodQuote, *, confirmation_name: str) -> dict[str, Any]:
        if confirmation_name != config.run_name:
            raise ValueError("typed confirmation must exactly match the run name")
        if not quote.available or quote.maximum_cost_usd > config.max_total_budget_usd:
            raise ValueError("quote exceeds availability or budget limits")
        return self.transport.create_pod({
            "name": config.run_name,
            "image": f"{config.image}@{config.image_digest}",
            "cloud": "SECURE",
            "startSsh": True,
            "ports": ["22/tcp"],
            "stopAfterSeconds": int(config.max_runtime_hours * 3600),
            "gpu": {"id": quote.gpu_type_id, "count": quote.gpu_count,
                    "minRamPerGpu": config.gpu_memory_gb},
            "disk": 200,
            **({"mounts": {"network": [{
                "volumeId": config.network_volume_id, "path": "/runpod-volume",
            }]}} if config.network_volume_id else {}),
        })

    def enforce_limits(self, pod_id: str, config: RunPodConfig, quote: RunPodQuote, *, elapsed_hours: float) -> dict[str, Any] | None:
        accrued = elapsed_hours * quote.hourly_price_usd
        if elapsed_hours >= config.max_runtime_hours or accrued >= config.max_total_budget_usd:
            return self.transport.stop_pod(pod_id)
        return None

    def cancel(self, pod_id: str) -> dict[str, Any]:
        return self.transport.stop_pod(pod_id)

    def reconnect(self, pod_id: str) -> dict[str, Any]:
        """Refresh remote state after control-plane restart."""
        return self.transport.get_pod(pod_id)

    @staticmethod
    def stage(executor: RunPodRemoteExecutor, experiment: GPUWorkerExperimentV1) -> str:
        root = f"/workspace/runs/{_safe_run_id(experiment.experiment_id)}"
        expected_output = root + "/output"
        expected_cancel = root + "/cancel.requested"
        if experiment.output_dir != expected_output:
            raise ValueError(f"RunPod experiment output_dir must be {expected_output}")
        if experiment.cancellation_file != expected_cancel:
            raise ValueError(f"RunPod experiment cancellation_file must be {expected_cancel}")
        executor.mkdir(root)
        executor.mkdir(root + "/inputs")
        staged_values: dict[str, str] = {}
        staged_manifest: list[dict[str, Any]] = []
        for field in ("environment_manifest", "rollout_dataset_path", "validation_dataset_path"):
            value = getattr(experiment, field)
            if value is None or _is_remote_resource(value):
                continue
            source = Path(value)
            if not source.is_file():
                raise ValueError(f"RunPod staging input is not a readable file: {field}")
            payload = source.read_bytes()
            suffix = "".join(source.suffixes)[-32:]
            remote_path = f"{root}/inputs/{field}{suffix}"
            executor.upload_bytes(remote_path, payload)
            staged_values[field] = remote_path
            staged_manifest.append({
                "field": field, "path": remote_path, "size_bytes": len(payload),
                "sha256": sha256(payload).hexdigest(),
            })
        for field in ("model_path", "caller_model_path", "verl_checkout", "tau2_checkout"):
            value = getattr(experiment, field)
            if value is not None and not _is_remote_resource(value):
                raise ValueError(
                    f"RunPod {field} must already exist under /workspace or /runpod-volume"
                )
        staged_experiment = experiment.model_copy(update=staged_values)
        destination = root + "/experiment.json"
        executor.upload_bytes(destination, staged_experiment.model_dump_json(indent=2).encode())
        executor.upload_bytes(
            root + "/staging-manifest.json",
            json.dumps(staged_manifest, sort_keys=True, separators=(",", ":")).encode(),
        )
        return destination

    @staticmethod
    def launch(executor: RunPodRemoteExecutor, experiment_path: str) -> dict[str, str]:
        root = str(Path(experiment_path).parent)
        if not experiment_path.startswith("/workspace/runs/") or not experiment_path.endswith("/experiment.json"):
            raise ValueError("RunPod experiment path is outside the managed workspace")
        stdout_path, exit_path, pid_path = root + "/worker.stream.log", root + "/worker.exit", root + "/worker.pid"
        executor.start(
            ("audioenv-gpu-worker", "run", experiment_path), stdout_path=stdout_path,
            exit_path=exit_path, pid_path=pid_path,
        )
        return {"stdout_path": stdout_path, "exit_path": exit_path, "pid_path": pid_path}

    @staticmethod
    def stream_events(
        executor: RunPodRemoteExecutor, *, experiment_id: str, after_sequence: int = 0,
    ) -> tuple[GPUWorkerEventV1, ...]:
        experiment_id = _safe_run_id(experiment_id)
        path = f"/workspace/runs/{experiment_id}/output/events.jsonl"
        try:
            payload = executor.read_bytes(path)
        except FileNotFoundError:
            return ()
        events = []
        for line in payload.splitlines():
            if not line.strip():
                continue
            event = GPUWorkerEventV1.model_validate_json(line)
            if event.experiment_id != experiment_id:
                raise ValueError("RunPod event belongs to another experiment")
            if event.sequence > after_sequence:
                events.append(event)
        events.sort(key=lambda item: item.sequence)
        expected = after_sequence + 1
        for event in events:
            if event.sequence != expected:
                raise ValueError("RunPod event stream is not contiguous")
            expected += 1
        return tuple(events)

    @staticmethod
    def request_cancellation(executor: RunPodRemoteExecutor, experiment_id: str) -> str:
        path = f"/workspace/runs/{_safe_run_id(experiment_id)}/cancel.requested"
        executor.touch(path)
        return path

    @staticmethod
    def collect_artifacts(
        executor: RunPodRemoteExecutor, *, experiment_id: str, destination: Path,
    ) -> tuple[Path, ...]:
        root = f"/workspace/runs/{_safe_run_id(experiment_id)}/output"
        manifest = json.loads(executor.read_bytes(root + "/worker-artifacts.json"))
        if not isinstance(manifest, list):
            raise ValueError("RunPod artifact manifest must be a list")
        destination.mkdir(parents=True, exist_ok=True)
        collected = []
        for row in manifest:
            relative = Path(str(row.get("path", "")))
            if relative.is_absolute() or ".." in relative.parts or not relative.parts:
                raise ValueError("RunPod artifact manifest contains an unsafe path")
            local = (destination / relative).resolve()
            if destination.resolve() not in local.parents:
                raise ValueError("RunPod artifact path escapes the destination")
            local.parent.mkdir(parents=True, exist_ok=True)
            executor.download(root + "/" + relative.as_posix(), local)
            payload = local.read_bytes()
            if len(payload) != int(row.get("size_bytes", -1)) or sha256(payload).hexdigest() != row.get("sha256"):
                raise ValueError(f"RunPod artifact integrity failed: {relative}")
            collected.append(local)
        return tuple(collected)

    def cleanup(self, pod_id: str) -> dict[str, Any]:
        """Stop compute while deliberately preserving any attached volume."""
        return self.transport.stop_pod(pod_id)

    @staticmethod
    def artifact_collection_command(
        *, host: str, remote_dir: str, local_dir: Path, ssh_key_path: Path,
    ) -> tuple[str, ...]:
        if not host or not remote_dir.startswith("/"):
            raise ValueError("absolute remote artifact directory and host are required")
        local_dir.mkdir(parents=True, exist_ok=True)
        return (
            "sftp", "-i", str(ssh_key_path), "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=yes", f"root@{host}:{remote_dir}/*", str(local_dir.resolve()),
        )


def _gpu_price(item: dict[str, Any]) -> float:
    value = item.get("price_per_gpu_hour")
    if value is None:
        value = item.get("pricePerGpuHour")
    if value is None:
        raise ValueError("GPU catalog item lacks an hourly price")
    return float(value)


def _safe_run_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", value):
        raise ValueError("experiment id is unsafe for a managed remote path")
    return value


def _managed_remote_path(value: str) -> str:
    path = Path(value)
    if (
        not value.startswith("/workspace/runs/")
        or ".." in path.parts
        or not re.fullmatch(r"/workspace/runs/[A-Za-z0-9._/-]+", value)
    ):
        raise ValueError("remote path is outside the managed RunPod workspace")
    return value


def _is_remote_resource(value: str) -> bool:
    path = Path(value)
    return (
        (value.startswith("/workspace/") or value.startswith("/runpod-volume/"))
        and ".." not in path.parts
        and bool(re.fullmatch(r"/[A-Za-z0-9._/-]+", value))
    )
