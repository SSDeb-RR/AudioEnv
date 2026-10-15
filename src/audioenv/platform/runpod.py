"""Guarded RunPod Secure Cloud provider contract.

No network call occurs at import, application startup, or preflight. A caller
must inject a v2 API transport, request a quote, satisfy budget/runtime limits,
and repeat the exact run name before provisioning can occur.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Protocol
from urllib.request import Request, urlopen


class RunPodTransport(Protocol):
    def list_gpu_types(self) -> list[dict[str, Any]]: ...
    def create_pod(self, payload: dict[str, Any]) -> dict[str, Any]: ...
    def get_pod(self, pod_id: str) -> dict[str, Any]: ...
    def stop_pod(self, pod_id: str) -> dict[str, Any]: ...


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
    gpu_count: int = 8
    gpu_memory_gb: int = 80
    max_runtime_hours: float = 8.0
    max_total_budget_usd: float = 200.0
    network_volume_id: str | None = None
    ssh_key_secret_name: str = "RUNPOD_SSH_KEY"
    api_key_secret_name: str = "RUNPOD_API_KEY"


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
            "image": config.image,
            "cloud": "SECURE",
            "startSsh": True,
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
