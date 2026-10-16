from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from audioenv.contracts import GPUWorkerEventV1, GPUWorkerExperimentV1
from audioenv.platform.api import _resolve_runpod_executor, create_app
from audioenv.platform.runpod import RunPodConfig, RunPodRESTV2Transport, RunPodSecureCloudProvider


class FakeTransport:
    def __init__(self):
        self.created = []
        self.stopped = []

    def list_gpu_types(self):
        return [{
            "id": "H100-80GB", "memory": 80,
            "maxCount": {"secure": 8}, "price": {"secure": 2.0},
        }]

    def create_pod(self, payload):
        self.created.append(payload)
        return {"id": "pod-1", "status": "RUNNING"}

    def get_pod(self, pod_id):
        return {"id": pod_id, "status": "RUNNING"}

    def stop_pod(self, pod_id):
        self.stopped.append(pod_id)
        return {"id": pod_id, "status": "STOPPED"}


class FakeRemote:
    def __init__(self):
        self.files: dict[str, bytes] = {}
        self.directories = []
        self.starts = []

    def mkdir(self, path):
        self.directories.append(path)

    def upload_bytes(self, path, payload):
        self.files[path] = payload

    def start(self, command, *, stdout_path, exit_path, pid_path):
        self.starts.append((command, stdout_path, exit_path, pid_path))
        self.files[pid_path] = b"123\n"

    def read_bytes(self, path):
        if path not in self.files:
            raise FileNotFoundError(path)
        return self.files[path]

    def touch(self, path):
        self.files[path] = b""

    def download(self, remote_path, local_path):
        local_path.write_bytes(self.files[remote_path])


def test_runpod_transport_uses_canonical_v2_catalog_with_live_pod_capacity(monkeypatch):
    transport = RunPodRESTV2Transport()
    calls = []
    monkeypatch.setattr(
        transport, "_request",
        lambda method, path, payload=None: calls.append((method, path, payload)) or {"gpus": []},
    )

    assert transport.base_url == "https://api.runpod.io"
    assert transport.list_gpu_types() == []
    assert calls == [(
        "GET",
        "/v2/catalog/gpus?include=AVAILABILITY&product=POD&gpuCount=8&cloud=SECURE",
        None,
    )]


def test_runpod_executor_resolves_v2_nested_direct_then_proxy_ssh(tmp_path, monkeypatch):
    key = tmp_path / "id_ed25519"; key.write_text("reference only")
    known = tmp_path / "known_hosts"; known.write_text("host key")
    monkeypatch.setenv("RUNPOD_SSH_KEY", str(key))
    monkeypatch.setenv("RUNPOD_KNOWN_HOSTS", str(known))
    app = SimpleNamespace(state=SimpleNamespace(runpod_executor_factory=None))
    config = RunPodConfig(
        run_name="proof", image_digest="sha256:" + "a" * 64,
    )
    pod = {"ssh": {
        "direct": {"host": "195.26.233.3", "port": 34446, "username": "root"},
        "proxy": {"host": "ssh.runpod.io", "port": 22, "username": "pod-123"},
    }}

    executor = _resolve_runpod_executor(app, pod, config, {})

    assert executor.host == "195.26.233.3"
    assert executor.port == 34446
    assert executor.user == "root"

    proxy = _resolve_runpod_executor(app, {"ssh": {
        "direct": None,
        "proxy": {"host": "ssh.runpod.io", "port": 22, "username": "pod-123"},
    }}, config, {})
    assert proxy.host == "ssh.runpod.io"
    assert proxy.port == 22
    assert proxy.user == "pod-123"


def experiment() -> GPUWorkerExperimentV1:
    return GPUWorkerExperimentV1(
        experiment_id="exp-1", run_name="gpu-proof", backend="tiny-cpu-proof",
        trainer="audioenv/small-audio-reinforce", algorithm="reinforce",
        output_dir="/workspace/runs/exp-1/output",
        cancellation_file="/workspace/runs/exp-1/cancel.requested",
    )


def test_runpod_remote_lifecycle_and_hash_verified_collection(tmp_path):
    transport = FakeTransport()
    provider = RunPodSecureCloudProvider(transport)
    remote = FakeRemote()
    bundle = experiment()

    staged = provider.stage(remote, bundle)
    assert staged == "/workspace/runs/exp-1/experiment.json"
    assert json.loads(remote.files[staged])["experiment_id"] == "exp-1"
    launch = provider.launch(remote, staged)
    assert remote.starts[0][0] == ("audioenv-gpu-worker", "run", staged)
    assert launch["pid_path"] == "/workspace/runs/exp-1/worker.pid"

    events = [
        GPUWorkerEventV1(
            sequence=index, experiment_id="exp-1", type=event_type,
            timestamp="2026-10-16T00:00:00Z", payload={},
        ).model_dump_json()
        for index, event_type in ((1, "worker.started"), (2, "worker.completed"))
    ]
    remote.files["/workspace/runs/exp-1/output/events.jsonl"] = ("\n".join(events) + "\n").encode()
    assert [item.sequence for item in provider.stream_events(remote, experiment_id="exp-1")] == [1, 2]
    assert [item.sequence for item in provider.stream_events(
        remote, experiment_id="exp-1", after_sequence=1,
    )] == [2]

    assert provider.request_cancellation(remote, "exp-1").endswith("cancel.requested")
    assert "/workspace/runs/exp-1/cancel.requested" in remote.files

    checkpoint = b"adapter checkpoint"
    artifact_path = "/workspace/runs/exp-1/output/checkpoints/adapter.bin"
    remote.files[artifact_path] = checkpoint
    remote.files["/workspace/runs/exp-1/output/worker-artifacts.json"] = json.dumps([{
        "path": "checkpoints/adapter.bin", "size_bytes": len(checkpoint),
        "sha256": sha256(checkpoint).hexdigest(),
    }]).encode()
    collected = provider.collect_artifacts(remote, experiment_id="exp-1", destination=tmp_path)
    assert collected == (tmp_path / "checkpoints/adapter.bin",)
    assert collected[0].read_bytes() == checkpoint

    assert provider.cleanup("pod-1")["status"] == "STOPPED"
    assert transport.stopped == ["pod-1"]


def test_runpod_rejects_mutable_images_unsafe_paths_and_event_gaps(tmp_path):
    with pytest.raises(ValueError, match="immutable"):
        RunPodConfig(run_name="unsafe")

    provider = RunPodSecureCloudProvider(FakeTransport())
    remote = FakeRemote()
    unsafe = experiment().model_copy(update={
        "experiment_id": "../escape", "output_dir": "/workspace/runs/../escape/output",
        "cancellation_file": "/workspace/runs/../escape/cancel.requested",
    })
    with pytest.raises(ValueError, match="unsafe"):
        provider.stage(remote, unsafe)

    event = GPUWorkerEventV1(
        sequence=2, experiment_id="exp-1", type="worker.started",
        timestamp="2026-10-16T00:00:00Z", payload={},
    )
    remote.files["/workspace/runs/exp-1/output/events.jsonl"] = (event.model_dump_json() + "\n").encode()
    with pytest.raises(ValueError, match="not contiguous"):
        provider.stream_events(remote, experiment_id="exp-1")

    payload = b"bad path"
    remote.files["/workspace/runs/exp-1/output/worker-artifacts.json"] = json.dumps([{
        "path": "../escape", "size_bytes": len(payload), "sha256": sha256(payload).hexdigest(),
    }]).encode()
    with pytest.raises(ValueError, match="unsafe"):
        provider.collect_artifacts(remote, experiment_id="exp-1", destination=tmp_path)


def test_runpod_stages_local_small_inputs_and_requires_remote_large_resources(tmp_path):
    provider = RunPodSecureCloudProvider(FakeTransport())
    remote = FakeRemote()
    manifest = tmp_path / "environment.json"
    rollouts = tmp_path / "rollouts.jsonl"
    validation = tmp_path / "validation.jsonl"
    manifest.write_text("{}")
    rollouts.write_text('{"row":1}\n')
    validation.write_text('{"row":2}\n')
    bundle = GPUWorkerExperimentV1(
        experiment_id="exp-qwen", run_name="gpu-proof", backend="qwen3-omni-verl",
        trainer="qwen3-omni-thinker-gspo-lora", algorithm="gspo",
        environment_manifest=str(manifest), rollout_dataset_path=str(rollouts),
        validation_dataset_path=str(validation), model_path="/runpod-volume/models/qwen",
        caller_model_path="/runpod-volume/models/qwen", verl_checkout="/workspace/verl-omni",
        tau2_checkout="/workspace/tau2", output_dir="/workspace/runs/exp-qwen/output",
        cancellation_file="/workspace/runs/exp-qwen/cancel.requested",
    )
    staged = provider.stage(remote, bundle)
    staged_payload = json.loads(remote.files[staged])
    assert staged_payload["environment_manifest"].startswith("/workspace/runs/exp-qwen/inputs/")
    assert staged_payload["model_path"] == "/runpod-volume/models/qwen"
    staging_manifest = json.loads(remote.files["/workspace/runs/exp-qwen/staging-manifest.json"])
    assert {item["field"] for item in staging_manifest} == {
        "environment_manifest", "rollout_dataset_path", "validation_dataset_path",
    }
    assert all(len(item["sha256"]) == 64 for item in staging_manifest)

    invalid = bundle.model_copy(update={"model_path": str(tmp_path / "model")})
    with pytest.raises(ValueError, match="model_path must already exist"):
        provider.stage(FakeRemote(), invalid)


def test_runpod_quote_review_and_exact_confirmation_are_persisted_product_flow(tmp_path):
    transport = FakeTransport()
    remote = FakeRemote()
    app = create_app(
        f"sqlite:///{tmp_path / 'product.sqlite'}", runpod_transport=transport,
        runpod_executor_factory=lambda _pod, _config: remote,
    )
    client = TestClient(app)
    project = client.get("/api/projects").json()[0]
    environment = client.get("/api/environments").json()[0]
    model_id = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    assert client.post("/api/models", json={
        "id": model_id, "display_name": "Qwen3 Omni", "adapter_type": "qwen3-omni",
        "role": "both", "project_id": project["id"],
        "config": {"checkpoint": "/runpod-volume/models/qwen"},
    }).status_code == 201
    experiment_response = client.post("/api/experiments", json={
        "name": "Paid Qwen proof", "project_id": project["id"], "config": {
            "project_id": project["id"], "environment_id": environment["id"],
            "agent_model_id": model_id, "caller_model_id": model_id,
            "trainer_id": "qwen3-omni-thinker-gspo-lora",
            "provider_id": "runpod-secure-cloud", "reward_version": "tau2-v1",
            "gpu_worker": {
                "environment_manifest": "/runpod-volume/tau2/environment.json",
                "model_path": "/runpod-volume/models/qwen",
                "caller_model_path": "/runpod-volume/models/qwen",
                "rollout_dataset_path": "/runpod-volume/tau2/train.jsonl",
                "validation_dataset_path": "/runpod-volume/tau2/test.jsonl",
                "verl_checkout": "/workspace/verl-omni", "tau2_checkout": "/workspace/tau2",
            },
        },
    })
    assert experiment_response.status_code == 201
    experiment_id = experiment_response.json()["id"]
    quote_response = client.post(f"/api/experiments/{experiment_id}/runpod/quote", json={
        "image": "registry.example/audioenv-gpu:v1",
        "image_digest": "sha256:" + "a" * 64,
        "max_runtime_hours": 2, "max_total_budget_usd": 40,
        "network_volume_id": "volume-1",
    })
    assert quote_response.status_code == 201, quote_response.text
    quote = quote_response.json()
    assert quote["status"] == "quoted"
    assert quote["config"]["quote"]["maximum_cost_usd"] == 32
    assert "RUNPOD_API_KEY" in quote_response.text
    assert "secret-value" not in quote_response.text
    assert transport.created == []

    wrong = client.post(f"/api/experiments/{experiment_id}/launch", json={
        "confirmed": True, "provider": "runpod-secure-cloud",
        "quote_id": quote["id"], "confirmation_name": "wrong",
    })
    assert wrong.status_code == 400
    assert transport.created == []

    launched = client.post(f"/api/experiments/{experiment_id}/launch", json={
        "confirmed": True, "provider": "runpod-secure-cloud",
        "quote_id": quote["id"], "confirmation_name": "Paid Qwen proof",
    })
    assert launched.status_code == 202, launched.text
    assert launched.json()["resource"]["status"] == "running"
    assert launched.json()["resource"]["external_id"] == "pod-1"
    assert transport.created[0]["image"].endswith("@sha256:" + "a" * 64)
    resources = client.get(f"/api/experiments/{experiment_id}/compute-resources").json()
    assert resources[0]["status"] == "running"
    assert client.get(f"/api/experiments/{experiment_id}").json()["status"] == "queued"
    reconciled = client.post(
        f"/api/experiments/{experiment_id}/compute-resources/{quote['id']}/reconcile",
    )
    assert reconciled.status_code == 200
    assert reconciled.json()["status"] == "running"
    gpu_events = [
        GPUWorkerEventV1(
            sequence=1, experiment_id=experiment_id, type="worker.started",
            timestamp="2026-10-16T00:00:00Z", payload={"backend": "qwen3-omni-verl"},
        ).model_dump_json(),
        GPUWorkerEventV1(
            sequence=2, experiment_id=experiment_id, type="metric",
            timestamp="2026-10-16T00:00:01Z",
            payload={"step": 1, "name": "loss", "value": 0.5, "source": "verl"},
        ).model_dump_json(),
    ]
    remote.files[f"/workspace/runs/{experiment_id}/output/events.jsonl"] = (
        "\n".join(gpu_events) + "\n"
    ).encode()
    synced = client.post(
        f"/api/experiments/{experiment_id}/compute-resources/{quote['id']}/sync",
    )
    assert synced.status_code == 200, synced.text
    assert synced.json()["events_synced"] == 2
    assert client.get(f"/api/experiments/{experiment_id}/metrics").json()[0]["value"] == 0.5
    assert client.get(f"/api/experiments/{experiment_id}").json()["status"] == "running"
    resource = app.state.repository.get_compute_resource(quote["id"])
    snapshot = json.loads(resource.config_json)
    snapshot["provisioned_at"] = "2000-01-01T00:00:00+00:00"
    app.state.repository.update_compute_resource(resource.id, status="running", config=snapshot)
    limited = client.post(
        f"/api/experiments/{experiment_id}/compute-resources/{quote['id']}/sync",
    )
    assert limited.status_code == 200, limited.text
    assert limited.json()["resource"]["status"] == "cancelled"
    assert client.get(f"/api/experiments/{experiment_id}").json()["status"] == "cancelled"
    assert f"/workspace/runs/{experiment_id}/cancel.requested" in remote.files
    assert transport.stopped == ["pod-1"]


def test_runpod_launch_rejects_changed_quote_before_provisioning(tmp_path):
    transport = FakeTransport()
    client = TestClient(create_app(
        f"sqlite:///{tmp_path / 'requote.sqlite'}", runpod_transport=transport,
    ))
    project = client.get("/api/projects").json()[0]
    environment = client.get("/api/environments").json()[0]
    model_id = "local-qwen"
    client.post("/api/models", json={
        "id": model_id, "display_name": "Qwen", "adapter_type": "qwen3-omni",
        "role": "both", "config": {"checkpoint": "/runpod-volume/qwen"},
    })
    experiment = client.post("/api/experiments", json={
        "name": "Requote proof", "project_id": project["id"], "config": {
            "environment_id": environment["id"], "agent_model_id": model_id,
            "trainer_id": "qwen3-omni-thinker-gspo-lora",
            "provider_id": "runpod-secure-cloud",
            "gpu_worker": {
                "environment_manifest": "/runpod-volume/tau2/environment.json",
                "model_path": "/runpod-volume/models/qwen",
                "caller_model_path": "/runpod-volume/models/qwen",
                "rollout_dataset_path": "/runpod-volume/tau2/train.jsonl",
                "validation_dataset_path": "/runpod-volume/tau2/test.jsonl",
                "verl_checkout": "/workspace/verl-omni", "tau2_checkout": "/workspace/tau2",
            },
        },
    }).json()
    quote = client.post(f"/api/experiments/{experiment['id']}/runpod/quote", json={
        "image": "registry/audioenv:v1", "image_digest": "sha256:" + "b" * 64,
        "max_runtime_hours": 2, "max_total_budget_usd": 100,
    }).json()
    transport.list_gpu_types = lambda: [{
        "id": "H100-80GB", "memory": 80, "maxCount": {"secure": 8},
        "price": {"secure": 2.5},
    }]
    response = client.post(f"/api/experiments/{experiment['id']}/launch", json={
        "confirmed": True, "provider": "runpod-secure-cloud", "quote_id": quote["id"],
        "confirmation_name": "Requote proof",
    })
    assert response.status_code == 409
    assert transport.created == []
    resources = client.get(f"/api/experiments/{experiment['id']}/compute-resources").json()
    assert resources[0]["status"] == "requote_required"


def test_runpod_terminal_sync_collects_artifacts_completes_and_stops_compute(tmp_path):
    transport = FakeTransport()
    remote = FakeRemote()
    app = create_app(
        f"sqlite:///{tmp_path / 'terminal.sqlite'}", runpod_transport=transport,
        runpod_executor_factory=lambda _pod, _config: remote,
    )
    repository = app.state.repository
    run = repository.create_experiment("Terminal GPU run", {"provider_id": "runpod-secure-cloud"})
    repository.transition(run.id, "validating")
    repository.transition(run.id, "queued")
    bundle = GPUWorkerExperimentV1(
        experiment_id=run.id, run_name=run.name, backend="tiny-cpu-proof",
        trainer="audioenv/small-audio-reinforce", algorithm="reinforce",
        output_dir=f"/workspace/runs/{run.id}/output",
        cancellation_file=f"/workspace/runs/{run.id}/cancel.requested",
    )
    config = RunPodConfig(
        run_name=run.name, image_digest="sha256:" + "c" * 64,
        max_runtime_hours=1, max_total_budget_usd=20,
    )
    resource = repository.create_compute_resource(
        experiment_id=run.id, provider="runpod-secure-cloud", status="running",
        external_id="pod-1", config={
            "config": asdict(config), "worker_experiment": bundle.model_dump(mode="json"),
            "worker_launch": {"experiment_path": f"/workspace/runs/{run.id}/experiment.json"},
        },
    )
    checkpoint = b"trained adapter"
    root = f"/workspace/runs/{run.id}/output"
    remote.files[f"{root}/events.jsonl"] = (
        GPUWorkerEventV1(
            sequence=1, experiment_id=run.id, type="worker.completed",
            timestamp="2026-10-16T00:00:00Z", payload={"artifact_count": 1},
        ).model_dump_json() + "\n"
    ).encode()
    remote.files[f"{root}/checkpoints/adapter.bin"] = checkpoint
    remote.files[f"{root}/worker-artifacts.json"] = json.dumps([{
        "path": "checkpoints/adapter.bin", "size_bytes": len(checkpoint),
        "sha256": sha256(checkpoint).hexdigest(),
    }]).encode()

    client = TestClient(app)
    response = client.post(
        f"/api/experiments/{run.id}/compute-resources/{resource.id}/sync",
    )
    assert response.status_code == 200, response.text
    assert response.json()["events_synced"] == 1
    assert len(response.json()["artifact_ids"]) == 1
    assert response.json()["resource"]["status"] == "stopped"
    assert client.get(f"/api/experiments/{run.id}").json()["status"] == "completed"
    artifacts = client.get("/api/artifacts").json()
    assert artifacts[0]["kind"] == "checkpoint"
    assert client.get(artifacts[0]["download_url"]).content == checkpoint
    assert transport.stopped == ["pod-1"]
