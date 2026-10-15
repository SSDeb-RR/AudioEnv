from hashlib import sha256
import json
from pathlib import Path

import pytest

from audioenv.contracts import GPUWorkerEventV1, GPUWorkerExperimentV1
from audioenv.platform.modal import ModalConfig, ModalSandboxProvider


class _Transport:
    def __init__(self):
        self.files = {}
        self.created = []
        self.terminated = []
        self.events = []

    def gpu_price(self, gpu_type, gpu_count):
        return {"available": True, "hourly_price_usd": 16.0}

    def put_volume_file(self, volume_name, remote_path, payload): self.files[(volume_name, remote_path)] = payload
    def get_volume_file(self, volume_name, remote_path): return self.files[(volume_name, remote_path)]
    def create_sandbox(self, payload): self.created.append(payload); return {"id": "sb-1", "state": "started"}
    def get_sandbox(self, sandbox_id): return {"id": sandbox_id, "state": "started"}
    def terminate_sandbox(self, sandbox_id): self.terminated.append(sandbox_id); return {"id": sandbox_id, "state": "finished"}
    def stream_stdout(self, sandbox_id, after_sequence): return self.events


def _config():
    return ModalConfig(
        run_name="qwen-proof", image="registry/audioenv-gpu:v1",
        image_digest="sha256:" + "a" * 64, max_runtime_hours=2,
        max_total_budget_usd=40, secret_names=("HF_TOKEN",),
    )


def _experiment(tmp_path):
    required = tmp_path / "required"
    required.mkdir()
    data = tmp_path / "rollouts.parquet"; data.write_bytes(b"data")
    manifest = tmp_path / "env.json"; manifest.write_text("{}")
    return GPUWorkerExperimentV1(
        experiment_id="exp-1", run_name="qwen-proof", backend="qwen3-omni-verl",
        trainer="qwen3-omni-thinker-gspo-lora", algorithm="gspo",
        environment_manifest=str(manifest), model_path=str(required), caller_model_path=str(required),
        rollout_dataset_path=str(data), verl_checkout=str(required), tau2_checkout=str(required),
        output_dir="/workspace/runs/exp-1/output",
    )


def test_modal_uses_same_worker_bundle_and_requires_fresh_exact_confirmation(tmp_path):
    transport = _Transport(); provider = ModalSandboxProvider(transport); config = _config()
    quote = provider.quote(config)
    assert quote.available and quote.maximum_cost_usd == 32
    path = provider.stage(config, _experiment(tmp_path))
    assert json.loads(transport.files[(config.volume_name, path)])["experiment_id"] == "exp-1"
    with pytest.raises(ValueError, match="exactly match"):
        provider.provision(config, quote, experiment_path=path, confirmation_name="wrong")
    assert not transport.created
    result = provider.provision(config, quote, experiment_path=path, confirmation_name="qwen-proof")
    assert result["id"] == "sb-1"
    payload = transport.created[0]
    assert payload["runtime"] == "gvisor"
    assert payload["secret_names"] == ["HF_TOKEN"]
    assert payload["command"] == ["audioenv-gpu-worker", "run", "/workspace/runs/exp-1/experiment.json"]


def test_modal_rejects_mutable_images_and_secret_values():
    with pytest.raises(ValueError, match="immutable"):
        ModalConfig(run_name="run", image="image:latest", image_digest="sha256:" + "z" * 64)
    with pytest.raises(ValueError, match="referenced by name"):
        ModalConfig(
            run_name="run", image="image:v1", image_digest="sha256:" + "a" * 64,
            secret_names=("HF_TOKEN=secret",),
        )


def test_modal_stream_reconnect_cancel_and_budget_stop():
    transport = _Transport(); provider = ModalSandboxProvider(transport); config = _config(); quote = provider.quote(config)
    transport.events = [GPUWorkerEventV1(
        sequence=1, experiment_id="exp-1", type="worker.started", timestamp="now", payload={},
    ).model_dump_json()]
    assert provider.stream_events("sb-1", experiment_id="exp-1")[0].sequence == 1
    assert provider.reconnect("sb-1")["state"] == "started"
    assert provider.enforce_limits("sb-1", config, quote, elapsed_hours=2)["state"] == "finished"
    provider.cancel("sb-2")
    assert transport.terminated == ["sb-1", "sb-2"]


def test_modal_collects_only_hash_verified_manifest_artifacts(tmp_path):
    transport = _Transport(); provider = ModalSandboxProvider(transport); config = _config()
    root = "/runs/exp-1/output"
    payload = b"checkpoint"
    manifest = [{"path": "checkpoints/adapter.bin", "size_bytes": len(payload), "sha256": sha256(payload).hexdigest()}]
    transport.files[(config.volume_name, f"{root}/worker-artifacts.json")] = json.dumps(manifest).encode()
    transport.files[(config.volume_name, f"{root}/checkpoints/adapter.bin")] = payload
    collected = provider.collect_artifacts(config, experiment_id="exp-1", destination=tmp_path / "out")
    assert collected[0].read_bytes() == payload
    manifest[0]["path"] = "../escape"
    transport.files[(config.volume_name, f"{root}/worker-artifacts.json")] = json.dumps(manifest).encode()
    with pytest.raises(ValueError, match="unsafe"):
        provider.collect_artifacts(config, experiment_id="exp-1", destination=tmp_path / "bad")
