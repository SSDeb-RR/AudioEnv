"""Guarded Modal Sandbox provider using the portable AudioEnv GPU worker.

The provider contains no implicit Modal import or network call. Production code
supplies an SDK-backed transport; tests use an in-memory transport. Secret names
are serialized, never their values, and provisioning requires an exact typed
confirmation after a quote.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Iterable, Protocol

from ..contracts import GPUWorkerEventV1, GPUWorkerExperimentV1


class ModalTransport(Protocol):
    def gpu_price(self, gpu_type: str, gpu_count: int) -> dict[str, Any]: ...
    def put_volume_file(self, volume_name: str, remote_path: str, payload: bytes) -> None: ...
    def create_sandbox(self, payload: dict[str, Any]) -> dict[str, Any]: ...
    def get_sandbox(self, sandbox_id: str) -> dict[str, Any]: ...
    def terminate_sandbox(self, sandbox_id: str) -> dict[str, Any]: ...
    def stream_stdout(self, sandbox_id: str, after_sequence: int) -> Iterable[str]: ...
    def get_volume_file(self, volume_name: str, remote_path: str) -> bytes: ...


class ModalSDKTransport:
    """Lazy adapter for Modal's official Python SDK.

    Importing AudioEnv never imports Modal. Constructing this transport is also
    side-effect free; network calls occur only in the explicit quote/stage/
    provision/reconnect methods invoked by the product workflow.
    """

    def __init__(
        self, *, app_name: str = "audioenv-gpu", hourly_prices: dict[str, float] | None = None,
        modal_module: Any | None = None,
    ):
        self.app_name = app_name
        self.hourly_prices = dict(hourly_prices or {})
        self._module = modal_module

    @property
    def modal(self):
        if self._module is None:
            try:
                import modal  # type: ignore[import-not-found]
            except ImportError as exc:
                raise RuntimeError("install AudioEnv with the 'modal' extra") from exc
            self._module = modal
        return self._module

    def gpu_price(self, gpu_type: str, gpu_count: int) -> dict[str, Any]:
        per_gpu = self.hourly_prices.get(gpu_type)
        return {
            "available": per_gpu is not None and per_gpu > 0,
            "hourly_price_usd": 0.0 if per_gpu is None else per_gpu * gpu_count,
            "basis": "operator-configured Modal rate; refreshed before launch",
        }

    def _volume(self, name: str):
        return self.modal.Volume.from_name(name, create_if_missing=True)

    def put_volume_file(self, volume_name: str, remote_path: str, payload: bytes) -> None:
        path = _volume_path(remote_path)
        with self._volume(volume_name).batch_upload(force=True) as batch:
            batch.put_file(BytesIO(payload), path)

    def get_volume_file(self, volume_name: str, remote_path: str) -> bytes:
        return b"".join(self._volume(volume_name).read_file(_volume_path(remote_path)))

    def create_sandbox(self, payload: dict[str, Any]) -> dict[str, Any]:
        modal = self.modal
        app = modal.App.lookup(self.app_name, create_if_missing=True)
        image = modal.Image.from_registry(f"{payload['image']}@{payload['image_digest']}")
        volume = self._volume(payload["volume"]["name"])
        secrets = [modal.Secret.from_name(name) for name in payload["secret_names"]]
        gpu = f"{payload['gpu']['type']}:{payload['gpu']['count']}"
        sandbox = modal.Sandbox.create(
            *payload["command"], app=app, image=image, gpu=gpu,
            runtime=payload["runtime"], timeout=payload["timeout_seconds"],
            volumes={payload["volume"]["mount_path"]: volume}, secrets=secrets,
        )
        return {"id": sandbox.object_id, "state": "started"}

    def _sandbox(self, sandbox_id: str):
        return self.modal.Sandbox.from_id(sandbox_id)

    def get_sandbox(self, sandbox_id: str) -> dict[str, Any]:
        sandbox = self._sandbox(sandbox_id)
        result = sandbox.poll()
        return {
            "id": sandbox_id, "state": "started" if result is None else "finished",
            "returncode": getattr(result, "returncode", result),
        }

    def terminate_sandbox(self, sandbox_id: str) -> dict[str, Any]:
        self._sandbox(sandbox_id).terminate()
        return {"id": sandbox_id, "state": "finished"}

    def stream_stdout(self, sandbox_id: str, after_sequence: int) -> Iterable[str]:
        for line in self._sandbox(sandbox_id).stdout:
            if isinstance(line, bytes):
                line = line.decode(errors="replace")
            if not line.strip():
                continue
            try:
                sequence = int(json.loads(line).get("sequence", 0))
            except (ValueError, TypeError, json.JSONDecodeError):
                continue
            if sequence > after_sequence:
                yield line


@dataclass(frozen=True, slots=True)
class ModalConfig:
    run_name: str
    image: str
    image_digest: str
    gpu_type: str = "H100"
    gpu_count: int = 8
    max_runtime_hours: float = 8.0
    max_total_budget_usd: float = 200.0
    volume_name: str = "audioenv-runs"
    secret_names: tuple[str, ...] = ()

    def __post_init__(self):
        if not self.run_name or not self.image:
            raise ValueError("run name and image are required")
        if re.fullmatch(r"sha256:[0-9a-f]{64}", self.image_digest) is None:
            raise ValueError("Modal image must be bound to an immutable sha256 digest")
        if self.gpu_count < 1 or self.max_runtime_hours <= 0 or self.max_runtime_hours > 24:
            raise ValueError("invalid Modal GPU count or runtime; Sandbox maximum is 24 hours")
        if self.max_total_budget_usd < 0:
            raise ValueError("budget cannot be negative")
        if not self.volume_name or len(set(self.secret_names)) != len(self.secret_names):
            raise ValueError("volume name is required and secret references must be unique")
        if any(not name or "=" in name for name in self.secret_names):
            raise ValueError("Modal secrets must be referenced by name, never serialized values")


@dataclass(frozen=True, slots=True)
class ModalQuote:
    gpu_type: str
    gpu_count: int
    hourly_price_usd: float
    maximum_cost_usd: float
    available: bool
    quote_id: str


class ModalSandboxProvider:
    provider_id = "modal-sandbox"

    def __init__(self, transport: ModalTransport):
        self.transport = transport

    def quote(self, config: ModalConfig) -> ModalQuote:
        result = self.transport.gpu_price(config.gpu_type, config.gpu_count)
        hourly = float(result.get("hourly_price_usd", 0.0))
        available = bool(result.get("available")) and hourly > 0
        maximum = hourly * config.max_runtime_hours
        quote_payload = {
            "gpu_type": config.gpu_type, "gpu_count": config.gpu_count,
            "hourly_price_usd": hourly, "maximum_cost_usd": maximum,
            "available": available and maximum <= config.max_total_budget_usd,
        }
        quote_id = sha256(json.dumps(quote_payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return ModalQuote(quote_id=quote_id, **quote_payload)

    def stage(self, config: ModalConfig, experiment: GPUWorkerExperimentV1) -> str:
        if experiment.run_name != config.run_name:
            raise ValueError("experiment run name does not match Modal configuration")
        root = f"/runs/{_safe_id(experiment.experiment_id)}"
        if experiment.output_dir != f"/workspace{root}/output":
            raise ValueError("Modal experiment output_dir must target the mounted run volume")
        if experiment.cancellation_file != f"/workspace{root}/cancel.requested":
            raise ValueError("Modal experiment cancellation_file must target the mounted run volume")
        staged_values: dict[str, str] = {}
        for field in ("environment_manifest", "rollout_dataset_path", "validation_dataset_path"):
            value = getattr(experiment, field)
            if value is None or value.startswith("/workspace/"):
                continue
            source = Path(value)
            if not source.is_file():
                raise ValueError(f"Modal staging input is not a readable file: {field}")
            suffix = "".join(source.suffixes)[-32:]
            volume_path = f"{root}/inputs/{field}{suffix}"
            self.transport.put_volume_file(config.volume_name, volume_path, source.read_bytes())
            staged_values[field] = f"/workspace{volume_path}"
        for field in ("model_path", "caller_model_path", "verl_checkout", "tau2_checkout"):
            value = getattr(experiment, field)
            if value is not None and not (
                value.startswith("/workspace/") or value.startswith("/opt/audioenv/checkouts/")
            ):
                raise ValueError(f"Modal {field} must already exist on the mounted volume or image")
        staged_experiment = experiment.model_copy(update=staged_values)
        remote_path = f"{root}/experiment.json"
        self.transport.put_volume_file(
            config.volume_name, remote_path,
            (staged_experiment.model_dump_json(indent=2) + "\n").encode(),
        )
        return remote_path

    def provision(
        self, config: ModalConfig, quote: ModalQuote, *, experiment_path: str,
        confirmation_name: str,
    ) -> dict[str, Any]:
        if confirmation_name != config.run_name:
            raise ValueError("typed confirmation must exactly match the run name")
        if not quote.available or quote.maximum_cost_usd > config.max_total_budget_usd:
            raise ValueError("quote exceeds availability or budget limits")
        expected = self.quote(config)
        if expected.quote_id != quote.quote_id:
            raise ValueError("Modal quote is stale or does not match configuration")
        if not experiment_path.startswith("/runs/") or not experiment_path.endswith("/experiment.json"):
            raise ValueError("staged experiment path is invalid")
        return self.transport.create_sandbox({
            "name": config.run_name,
            "image": config.image,
            "image_digest": config.image_digest,
            "gpu": {"type": config.gpu_type, "count": config.gpu_count},
            "runtime": "gvisor",
            "timeout_seconds": int(config.max_runtime_hours * 3600),
            "volume": {"name": config.volume_name, "mount_path": "/workspace"},
            "secret_names": list(config.secret_names),
            "command": ["audioenv-gpu-worker", "run", f"/workspace{experiment_path}"],
        })

    def stream_events(
        self, sandbox_id: str, *, experiment_id: str, after_sequence: int = 0,
    ) -> tuple[GPUWorkerEventV1, ...]:
        events: list[GPUWorkerEventV1] = []
        expected = after_sequence + 1
        for line in self.transport.stream_stdout(sandbox_id, after_sequence):
            event = GPUWorkerEventV1.model_validate_json(line)
            if event.experiment_id != experiment_id:
                raise ValueError("Modal event belongs to another experiment")
            if event.sequence != expected:
                raise ValueError("Modal event sequence is not contiguous")
            events.append(event)
            expected += 1
        return tuple(events)

    def collect_artifacts(
        self, config: ModalConfig, *, experiment_id: str, destination: Path,
    ) -> tuple[Path, ...]:
        root = PurePosixPath(f"/runs/{experiment_id}/output")
        manifest_path = str(root / "worker-artifacts.json")
        manifest_payload = self.transport.get_volume_file(config.volume_name, manifest_path)
        manifest = json.loads(manifest_payload)
        if not isinstance(manifest, list):
            raise ValueError("worker artifact manifest must be a list")
        destination.mkdir(parents=True, exist_ok=True)
        collected: list[Path] = []
        for item in manifest:
            relative = PurePosixPath(str(item["path"]))
            if relative.is_absolute() or ".." in relative.parts or relative.name == "worker-artifacts.json":
                raise ValueError("unsafe worker artifact path")
            payload = self.transport.get_volume_file(config.volume_name, str(root / relative))
            if len(payload) != int(item["size_bytes"]) or sha256(payload).hexdigest() != item["sha256"]:
                raise ValueError(f"artifact integrity check failed: {relative}")
            local = (destination / Path(*relative.parts)).resolve()
            if not local.is_relative_to(destination.resolve()):
                raise ValueError("artifact path escapes destination")
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(payload)
            collected.append(local)
        manifest_local = destination / "worker-artifacts.json"
        manifest_local.write_bytes(manifest_payload)
        return tuple(collected)

    def enforce_limits(
        self, sandbox_id: str, config: ModalConfig, quote: ModalQuote, *, elapsed_hours: float,
    ) -> dict[str, Any] | None:
        if elapsed_hours >= config.max_runtime_hours or elapsed_hours * quote.hourly_price_usd >= config.max_total_budget_usd:
            return self.transport.terminate_sandbox(sandbox_id)
        return None

    def reconnect(self, sandbox_id: str) -> dict[str, Any]:
        return self.transport.get_sandbox(sandbox_id)

    def cancel(self, sandbox_id: str) -> dict[str, Any]:
        return self.transport.terminate_sandbox(sandbox_id)


def _safe_id(value: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", value) is None:
        raise ValueError("unsafe Modal experiment id")
    return value


def _volume_path(value: str) -> str:
    path = PurePosixPath(value)
    if not value.startswith("/") or ".." in path.parts or re.fullmatch(r"/[A-Za-z0-9._/-]+", value) is None:
        raise ValueError("unsafe Modal volume path")
    return value
