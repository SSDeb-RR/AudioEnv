from hashlib import sha256
import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from audioenv.contracts import GPUExecutionPlanV1, GPUWorkerEventV1, GPUWorkerExperimentV1
from audioenv.platform.api import create_app
from audioenv.platform.gpu import GPUPreflightResult, LocalCUDAExecutionProvider


class FakeDocker:
    def __init__(self):
        self.commands: list[tuple[str, ...]] = []
        self.lines: list[str] = []
        self.stopped: list[str] = []
        self.removed: list[str] = []
        self.state = {"running": True, "status": "running", "exit_code": 0}

    def start(self, command):
        self.commands.append(command)
        return "a" * 64

    def inspect(self, container_id):
        return {"id": container_id, **self.state}

    def logs(self, container_id):
        return tuple(self.lines)

    def stop(self, container_id):
        self.stopped.append(container_id)

    def remove(self, container_id):
        self.removed.append(container_id)


def _plan():
    return GPUExecutionPlanV1(
        provider="local-cuda", image="registry.example/audioenv-gpu:v1",
        image_digest="sha256:" + "d" * 64, gpu_type="H100", gpu_count=4,
        memory_gb_per_gpu=80, storage_gb=500, max_runtime_hours=2,
        max_total_budget_usd=0,
    )


def _experiment(root: Path):
    inputs = root / "inputs"
    inputs.mkdir(parents=True)
    manifest = inputs / "environment.json"; manifest.write_text("{}")
    train = inputs / "train.jsonl"; train.write_text('{}\n')
    test = inputs / "test.jsonl"; test.write_text('{}\n')
    model = root / "models" / "qwen"; model.mkdir(parents=True)
    return GPUWorkerExperimentV1(
        experiment_id="exp-local", run_name="Local Qwen proof", backend="qwen3-omni-verl",
        trainer="qwen3-omni-thinker-gspo-lora", algorithm="gspo",
        environment_manifest=str(manifest), model_path=str(model), caller_model_path=str(model),
        rollout_dataset_path=str(train), validation_dataset_path=str(test),
        verl_checkout="/opt/audioenv/checkouts/verl-omni",
        tau2_checkout="/opt/audioenv/checkouts/tau2",
        output_dir="/workspace/runs/exp-local/output",
        cancellation_file="/workspace/runs/exp-local/cancel.requested",
    )


def test_local_cuda_stages_launches_streams_and_collects_same_worker_contract(tmp_path):
    docker = FakeDocker(); provider = LocalCUDAExecutionProvider(docker)
    experiment_path = provider.stage(_plan(), data_dir=tmp_path, experiment=_experiment(tmp_path))
    staged = json.loads(experiment_path.read_text())
    assert staged["model_path"] == "/workspace/models/qwen"
    assert staged["verl_checkout"] == "/opt/audioenv/checkouts/verl-omni"
    with pytest.raises(ValueError, match="exactly match"):
        provider.launch(
            _plan(), data_dir=tmp_path, experiment_path=experiment_path,
            run_name="Local Qwen proof", confirmation_name="wrong",
        )
    launched = provider.launch(
        _plan(), data_dir=tmp_path, experiment_path=experiment_path,
        run_name="Local Qwen proof", confirmation_name="Local Qwen proof",
    )
    assert launched["container_id"] == "a" * 64
    command = docker.commands[0]
    assert "--network=none" in command and "--read-only" in command
    assert f"--user={os.getuid()}:{os.getgid()}" in command
    assert f"registry.example/audioenv-gpu:v1@{'sha256:' + 'd' * 64}" in command
    assert command[-3:] == ("audioenv-gpu-worker", "run", "/workspace/runs/exp-local/experiment.json")

    docker.lines = [
        "non-json framework output",
        GPUWorkerEventV1(
            sequence=1, experiment_id="exp-local", type="worker.started",
            timestamp="now", payload={},
        ).model_dump_json(),
        GPUWorkerEventV1(
            sequence=2, experiment_id="exp-local", type="worker.completed",
            timestamp="now", payload={},
        ).model_dump_json(),
    ]
    assert [event.sequence for event in provider.stream_events(
        launched["container_id"], experiment_id="exp-local",
    )] == [1, 2]
    assert provider.reconnect(launched["container_id"])["running"]

    output = tmp_path / "runs" / "exp-local" / "output"
    checkpoint = output / "checkpoints" / "adapter.bin"
    checkpoint.parent.mkdir(parents=True); checkpoint.write_bytes(b"adapter")
    (output / "worker-artifacts.json").write_text(json.dumps([{
        "path": "checkpoints/adapter.bin", "size_bytes": 7,
        "sha256": sha256(b"adapter").hexdigest(),
    }]))
    collected = provider.collect_artifacts(
        data_dir=tmp_path, experiment_id="exp-local", destination=tmp_path / "collected",
    )
    assert collected[0].read_bytes() == b"adapter"
    provider.cancel(launched["container_id"])
    provider.cleanup(launched["container_id"])
    assert docker.stopped == ["a" * 64]
    assert docker.removed == ["a" * 64]


def test_local_cuda_rejects_resources_outside_mount_and_event_gaps(tmp_path):
    outside = tmp_path.parent / "outside-audioenv.json"
    outside.write_text("{}")
    experiment = _experiment(tmp_path).model_copy(update={"environment_manifest": str(outside)})
    provider = LocalCUDAExecutionProvider(FakeDocker())
    with pytest.raises(ValueError, match="contained"):
        provider.stage(_plan(), data_dir=tmp_path, experiment=experiment)
    outside.unlink()

    provider.transport.lines = [GPUWorkerEventV1(
        sequence=2, experiment_id="exp-local", type="worker.started",
        timestamp="now", payload={},
    ).model_dump_json()]
    with pytest.raises(ValueError, match="not contiguous"):
        provider.stream_events("a" * 64, experiment_id="exp-local")


class PassingLocalCUDA(LocalCUDAExecutionProvider):
    def preflight(self, plan, *, model_path, required_free_storage_gb=200):
        return GPUPreflightResult(True, (), {
            "provider": "local-cuda", "devices": [{"name": "Fake H100", "memory_mb": 81920}],
            "free_storage_gb": 500,
        })


def test_local_cuda_persisted_api_lifecycle_requires_preflight_and_confirmation(tmp_path):
    docker = FakeDocker(); provider = PassingLocalCUDA(docker)
    data_dir = tmp_path / "gpu-data"; data_dir.mkdir()
    inputs = data_dir / "tau2"; inputs.mkdir()
    for name in ("environment.json", "train.jsonl", "test.jsonl"):
        (inputs / name).write_text("{}\n")
    model = data_dir / "models" / "qwen"; model.mkdir(parents=True)
    client = TestClient(create_app(
        f"sqlite:///{tmp_path / 'platform.sqlite'}", local_cuda_provider=provider,
    ))
    project = client.get("/api/projects").json()[0]
    environment = client.get("/api/environments").json()[0]
    model_id = "local-qwen"
    assert client.post("/api/models", json={
        "id": model_id, "display_name": "Local Qwen", "adapter_type": "qwen3-omni",
        "role": "both", "project_id": project["id"], "config": {"checkpoint": str(model)},
    }).status_code == 201
    run = client.post("/api/experiments", json={
        "name": "Local CUDA proof", "project_id": project["id"], "config": {
            "project_id": project["id"], "environment_id": environment["id"],
            "agent_model_id": model_id, "caller_model_id": model_id,
            "trainer_id": "qwen3-omni-thinker-gspo-lora", "provider_id": "local-cuda",
            "gpu_worker": {
                "environment_manifest": str(inputs / "environment.json"),
                "model_path": str(model), "caller_model_path": str(model),
                "rollout_dataset_path": str(inputs / "train.jsonl"),
                "validation_dataset_path": str(inputs / "test.jsonl"),
                "verl_checkout": "/opt/audioenv/checkouts/verl-omni",
                "tau2_checkout": "/opt/audioenv/checkouts/tau2",
            },
        },
    }).json()
    plan = _plan().model_dump(mode="json")
    prepared_response = client.post(f"/api/experiments/{run['id']}/local-cuda/prepare", json={
        "plan": plan, "data_dir": str(data_dir), "model_path": str(model),
    })
    assert prepared_response.status_code == 201, prepared_response.text
    prepared = prepared_response.json()
    assert prepared["status"] == "prepared" and not docker.commands
    wrong = client.post(f"/api/experiments/{run['id']}/launch", json={
        "confirmed": True, "provider": "local-cuda", "compute_resource_id": prepared["id"],
        "confirmation_name": "wrong",
    })
    assert wrong.status_code == 400 and not docker.commands
    launched = client.post(f"/api/experiments/{run['id']}/launch", json={
        "confirmed": True, "provider": "local-cuda", "compute_resource_id": prepared["id"],
        "confirmation_name": "Local CUDA proof",
    })
    assert launched.status_code == 202, launched.text
    assert launched.json()["resource"]["status"] == "running"

    output = data_dir / "runs" / run["id"] / "output"
    checkpoint = output / "checkpoints" / "adapter.bin"
    checkpoint.parent.mkdir(parents=True); checkpoint.write_bytes(b"adapter")
    (output / "worker-artifacts.json").write_text(json.dumps([{
        "path": "checkpoints/adapter.bin", "size_bytes": 7,
        "sha256": sha256(b"adapter").hexdigest(),
    }]))
    docker.lines = [
        GPUWorkerEventV1(
            sequence=1, experiment_id=run["id"], type="worker.started", timestamp="now", payload={},
        ).model_dump_json(),
        GPUWorkerEventV1(
            sequence=2, experiment_id=run["id"], type="metric", timestamp="now",
            payload={"step": 1, "name": "loss", "value": 0.2, "source": "verl"},
        ).model_dump_json(),
        GPUWorkerEventV1(
            sequence=3, experiment_id=run["id"], type="worker.completed", timestamp="now", payload={},
        ).model_dump_json(),
    ]
    synced = client.post(
        f"/api/experiments/{run['id']}/compute-resources/{prepared['id']}/sync",
    )
    assert synced.status_code == 200, synced.text
    assert synced.json()["events_synced"] == 3
    assert client.get(f"/api/experiments/{run['id']}").json()["status"] == "completed"
    assert client.get(f"/api/experiments/{run['id']}/metrics").json()[0]["value"] == 0.2
    assert docker.stopped == ["a" * 64]
    assert docker.removed == ["a" * 64]
