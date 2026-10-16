from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from audioenv.contracts import GPUWorkerEventV1, GPUWorkerExperimentV1
from audioenv.platform.api import create_app
from audioenv.platform.modal import ModalConfig, ModalSDKTransport, ModalSandboxProvider


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
    validation = tmp_path / "validation.parquet"; validation.write_bytes(b"data")
    manifest = tmp_path / "env.json"; manifest.write_text("{}")
    return GPUWorkerExperimentV1(
        experiment_id="exp-1", run_name="qwen-proof", backend="qwen3-omni-verl",
        trainer="qwen3-omni-thinker-gspo-lora", algorithm="gspo",
        environment_manifest=str(manifest), model_path="/workspace/models/qwen", caller_model_path="/workspace/models/qwen",
        rollout_dataset_path=str(data), validation_dataset_path=str(validation),
        verl_checkout="/workspace/verl-omni", tau2_checkout="/workspace/tau2",
        output_dir="/workspace/runs/exp-1/output",
        cancellation_file="/workspace/runs/exp-1/cancel.requested",
    )


def test_modal_uses_same_worker_bundle_and_requires_fresh_exact_confirmation(tmp_path):
    transport = _Transport(); provider = ModalSandboxProvider(transport); config = _config()
    quote = provider.quote(config)
    assert quote.available and quote.maximum_cost_usd == 32
    path = provider.stage(config, _experiment(tmp_path))
    staged = json.loads(transport.files[(config.volume_name, path)])
    assert staged["experiment_id"] == "exp-1"
    assert staged["rollout_dataset_path"].startswith("/workspace/runs/exp-1/inputs/")
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


class _FakeUpload:
    def __init__(self, volume): self.volume = volume
    def __enter__(self): return self
    def __exit__(self, *_args): return False
    def put_file(self, source, destination): self.volume.files[destination] = source.read()


class _FakeVolume:
    def __init__(self): self.files = {}
    def batch_upload(self, force=False): assert force; return _FakeUpload(self)
    def read_file(self, path): yield self.files[path]


class _FakeSandbox:
    def __init__(self): self.object_id = "sb-sdk"; self.stdout = [] ; self.terminated = False
    def poll(self): return None
    def terminate(self): self.terminated = True


def test_modal_sdk_transport_maps_official_sdk_boundaries_without_network():
    volume = _FakeVolume(); sandbox = _FakeSandbox(); calls = {}
    fake = SimpleNamespace(
        App=SimpleNamespace(lookup=lambda name, create_if_missing: (name, create_if_missing)),
        Image=SimpleNamespace(from_registry=lambda image: ("image", image)),
        Volume=SimpleNamespace(from_name=lambda name, create_if_missing: volume),
        Secret=SimpleNamespace(from_name=lambda name: ("secret", name)),
        Sandbox=SimpleNamespace(
            create=lambda *command, **kwargs: calls.update(command=command, kwargs=kwargs) or sandbox,
            from_id=lambda sandbox_id: sandbox,
        ),
    )
    transport = ModalSDKTransport(
        modal_module=fake, hourly_prices={"H100": 2.5}, app_name="audioenv-test",
    )
    assert transport.gpu_price("H100", 8)["hourly_price_usd"] == 20
    transport.put_volume_file("runs", "/runs/e/experiment.json", b"{}")
    assert transport.get_volume_file("runs", "/runs/e/experiment.json") == b"{}"
    created = transport.create_sandbox({
        "image": "registry/audioenv:v1", "image_digest": "sha256:" + "a" * 64,
        "gpu": {"type": "H100", "count": 8}, "runtime": "gvisor",
        "timeout_seconds": 3600, "volume": {"name": "runs", "mount_path": "/workspace"},
        "secret_names": ["HF_TOKEN"], "command": ["audioenv-gpu-worker", "run", "/workspace/e.json"],
    })
    assert created == {"id": "sb-sdk", "state": "started"}
    assert calls["command"] == ("audioenv-gpu-worker", "run", "/workspace/e.json")
    assert calls["kwargs"]["gpu"] == "H100:8"
    assert calls["kwargs"]["runtime"] == "gvisor"
    sandbox.stdout = [b'{"sequence":1}\n']
    assert list(transport.stream_stdout("sb-sdk", 0)) == ['{"sequence":1}\n']
    assert list(transport.stream_stdout("sb-sdk", 1)) == []
    assert transport.get_sandbox("sb-sdk")["state"] == "started"
    assert transport.terminate_sandbox("sb-sdk")["state"] == "finished"
    assert sandbox.terminated


def test_modal_quote_launch_sync_and_cancel_are_persisted_product_flow(tmp_path):
    transport = _Transport()
    app = create_app(
        f"sqlite:///{tmp_path / 'modal.sqlite'}", modal_transport=transport,
    )
    client = TestClient(app)
    project = client.get("/api/projects").json()[0]
    environment = client.get("/api/environments").json()[0]
    model_id = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    assert client.post("/api/models", json={
        "id": model_id, "display_name": "Qwen3 Omni", "adapter_type": "qwen3-omni",
        "role": "both", "project_id": project["id"],
        "config": {"checkpoint": "/workspace/models/qwen"},
    }).status_code == 201
    experiment = client.post("/api/experiments", json={
        "name": "Modal Qwen proof", "project_id": project["id"], "config": {
            "project_id": project["id"], "environment_id": environment["id"],
            "agent_model_id": model_id, "caller_model_id": model_id,
            "trainer_id": "qwen3-omni-thinker-gspo-lora",
            "provider_id": "modal-sandbox", "reward_version": "tau2-v1",
            "gpu_worker": {
                "environment_manifest": "/workspace/tau2/environment.json",
                "model_path": "/workspace/models/qwen",
                "caller_model_path": "/workspace/models/qwen",
                "rollout_dataset_path": "/workspace/tau2/train.jsonl",
                "validation_dataset_path": "/workspace/tau2/test.jsonl",
                "verl_checkout": "/opt/audioenv/checkouts/verl-omni",
                "tau2_checkout": "/opt/audioenv/checkouts/tau2",
            },
        },
    }).json()
    preflight = client.post(f"/api/experiments/{experiment['id']}/preflight").json()
    assert preflight["available"] and preflight["provider_id"] == "modal-sandbox"
    quote_response = client.post(f"/api/experiments/{experiment['id']}/modal/quote", json={
        "image": "registry.example/audioenv-gpu:v1",
        "image_digest": "sha256:" + "a" * 64,
        "max_runtime_hours": 2, "max_total_budget_usd": 40,
        "hourly_price_per_gpu_usd": 2, "secret_names": ["HF_TOKEN"],
    })
    assert quote_response.status_code == 201, quote_response.text
    quote = quote_response.json()
    assert quote["config"]["pricing_basis"].startswith("operator-configured")
    assert not transport.created
    wrong = client.post(f"/api/experiments/{experiment['id']}/launch", json={
        "confirmed": True, "provider": "modal-sandbox",
        "quote_id": quote["id"], "confirmation_name": "wrong",
    })
    assert wrong.status_code == 400 and not transport.created
    launched = client.post(f"/api/experiments/{experiment['id']}/launch", json={
        "confirmed": True, "provider": "modal-sandbox",
        "quote_id": quote["id"], "confirmation_name": "Modal Qwen proof",
    })
    assert launched.status_code == 202, launched.text
    assert launched.json()["resource"]["external_id"] == "sb-1"
    transport.events = [GPUWorkerEventV1(
        sequence=1, experiment_id=experiment["id"], type="metric", timestamp="now",
        payload={"step": 1, "name": "loss", "value": 0.25, "source": "verl"},
    ).model_dump_json()]
    synced = client.post(
        f"/api/experiments/{experiment['id']}/compute-resources/{quote['id']}/sync",
    )
    assert synced.status_code == 200, synced.text
    assert client.get(f"/api/experiments/{experiment['id']}/metrics").json()[0]["value"] == 0.25
    resource = app.state.repository.get_compute_resource(quote["id"])
    snapshot = json.loads(resource.config_json)
    snapshot["provisioned_at"] = "2000-01-01T00:00:00+00:00"
    app.state.repository.update_compute_resource(resource.id, status="running", config=snapshot)
    limited = client.post(
        f"/api/experiments/{experiment['id']}/compute-resources/{quote['id']}/sync",
    )
    assert limited.status_code == 200, limited.text
    assert limited.json()["resource"]["status"] == "cancelled"
    assert client.get(f"/api/experiments/{experiment['id']}").json()["status"] == "cancelled"
    assert transport.terminated == ["sb-1"]
