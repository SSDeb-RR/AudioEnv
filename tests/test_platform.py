from datetime import timedelta
from hashlib import sha256
import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from audioenv.platform.api import (
    _enforce_resume_storage, _remote_resume_checkpoint, _resume_storage_binding,
    _store_gpu_artifact, create_app,
)
from audioenv.platform.db import (
    Database,
    InvalidTransition,
    JobRecord,
    PlatformRepository,
    utcnow,
)
from audioenv.platform.export import export_experiment


def test_experiment_lifecycle_and_invalid_transition(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'db.sqlite'}")
    database.create_schema()
    repository = PlatformRepository(database)
    experiment = repository.create_experiment("test", {"trainer": "tiny"})
    assert experiment.status == "draft"
    with pytest.raises(InvalidTransition):
        repository.transition(experiment.id, "running")
    for target in ("validating", "queued", "running", "evaluating", "completed"):
        experiment = repository.transition(experiment.id, target)
    assert experiment.status == "completed"


def test_gpu_resume_lineage_requires_preserved_checkpoint_directory(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'resume.sqlite'}")
    database.create_schema()
    repository = PlatformRepository(database)
    experiment = repository.create_experiment("gpu", {"provider_id": "local-cuda"})
    artifact = SimpleNamespace(metadata_json=json.dumps({
        "remote_relative_path": "validation/global_step_7/adapter/model.safetensors",
    }))
    assert _remote_resume_checkpoint(
        repository, experiment.id, "local-cuda", artifact,
    ) == f"/workspace/runs/{experiment.id}/output/validation/global_step_7"

    with pytest.raises(HTTPException, match="persistent network volume"):
        _remote_resume_checkpoint(repository, experiment.id, "runpod-secure-cloud", artifact)
    repository.create_compute_resource(
        experiment_id=experiment.id, provider="runpod-secure-cloud", status="stopped",
        config={"config": {"network_volume_id": "volume-1"}},
    )
    assert _remote_resume_checkpoint(
        repository, experiment.id, "runpod-secure-cloud", artifact,
    ).endswith("/validation/global_step_7")
    binding = _resume_storage_binding(
        repository, experiment.id, "runpod-secure-cloud",
    )
    assert binding["storage_id"] == "volume-1"
    _enforce_resume_storage({"resume_storage": binding}, "runpod-secure-cloud", "volume-1")
    with pytest.raises(HTTPException, match="original persisted storage"):
        _enforce_resume_storage(
            {"resume_storage": binding}, "runpod-secure-cloud", "different-volume",
        )


def test_gpu_checkpoint_verification_uses_matching_worker_reload_evidence(tmp_path):
    app = create_app(f"sqlite:///{tmp_path / 'gpu-checkpoint.sqlite'}")
    repository = app.state.repository
    experiment = repository.create_experiment("gpu", {"provider_id": "local-cuda"})
    repository.add_log(experiment.id, "checkpoint.saved", payload={
        "gpu_event_sequence": 19,
        "path": f"/workspace/runs/{experiment.id}/output/global_step_4",
        "sha256": "tree-digest",
        "reload_verified": True,
    })
    incoming = app.state.data_root / "incoming" / "resource-1"
    checkpoint = incoming / "global_step_4" / "adapter" / "model.safetensors"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"gpu adapter shard")

    artifact_id = _store_gpu_artifact(app, repository, experiment, checkpoint, incoming)
    artifact = repository.get_artifact(artifact_id)
    metadata = json.loads(artifact.metadata_json)
    assert metadata["verification_source"] == "gpu-worker-checkpoint-reload"
    assert metadata["checkpoint_tree_sha256"] == "tree-digest"
    assert metadata["step"] == 4

    response = TestClient(app).post(
        f"/api/experiments/{experiment.id}/checkpoints/{artifact_id}/verify",
    )
    assert response.status_code == 200
    assert response.json() == {
        "verified": True,
        "artifact_id": artifact_id,
        "source": "gpu-worker-checkpoint-reload",
        "checkpoint_directory": "global_step_4",
        "checkpoint_tree_sha256": "tree-digest",
        "step": 4,
    }


def test_gpu_checkpoint_verification_fails_closed_without_matching_reload_evidence(tmp_path):
    app = create_app(f"sqlite:///{tmp_path / 'unverified-gpu-checkpoint.sqlite'}")
    repository = app.state.repository
    experiment = repository.create_experiment("gpu", {"provider_id": "local-cuda"})
    repository.add_log(experiment.id, "checkpoint.saved", payload={
        "path": f"/workspace/runs/{experiment.id}/output/global_step_3",
        "sha256": "different-tree",
        "reload_verified": True,
    })
    incoming = app.state.data_root / "incoming" / "resource-2"
    checkpoint = incoming / "global_step_4" / "adapter" / "model.safetensors"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"not a tiny checkpoint")

    artifact_id = _store_gpu_artifact(app, repository, experiment, checkpoint, incoming)
    metadata = json.loads(repository.get_artifact(artifact_id).metadata_json)
    assert "verification_source" not in metadata
    response = TestClient(app).post(
        f"/api/experiments/{experiment.id}/checkpoints/{artifact_id}/verify",
    )
    assert response.status_code == 200
    assert response.json()["verified"] is False


def test_resumed_runpod_quote_rejects_a_different_volume_before_provisioning(tmp_path):
    app = create_app(f"sqlite:///{tmp_path / 'resume-api.sqlite'}")
    repository = app.state.repository
    source = repository.create_experiment("gpu source", {
        "provider_id": "runpod-secure-cloud",
        "trainer_id": "qwen3-omni-thinker-gspo-lora",
        "gpu_worker": {},
    })
    for target in ("validating", "queued", "running", "evaluating", "completed"):
        source = repository.transition(source.id, target)
    stored = app.state.artifacts.put_bytes(b"checkpoint", media_type="application/octet-stream")
    repository.add_artifact(
        stored.id, experiment_id=source.id, project_id=source.project_id,
        kind="checkpoint", media_type=stored.media_type, size_bytes=stored.size_bytes,
        path=str(stored.path), metadata={
            "remote_relative_path": "global_step_4/adapter/model.safetensors",
            "verification_source": "gpu-worker-checkpoint-reload",
            "checkpoint_reload_verified": True,
            "checkpoint_directory": "global_step_4",
        },
    )
    repository.create_compute_resource(
        experiment_id=source.id, provider="runpod-secure-cloud", status="stopped",
        config={"config": {"network_volume_id": "original-volume"}},
    )
    client = TestClient(app)
    resumed = client.post(f"/api/experiments/{source.id}/resume", json={
        "checkpoint_artifact_id": stored.id, "run_name": "gpu resumed",
        "additional_steps": 1,
    })
    assert resumed.status_code == 201
    resumed_run = resumed.json()
    assert resumed_run["status"] == "draft"
    assert resumed_run["config"]["hyperparameters"]["max_steps"] == 1
    assert resumed_run["config"]["gpu_worker"]["resume_from"].endswith("/global_step_4")
    assert resumed_run["config"]["resume_storage"]["provider_id"] == "runpod-secure-cloud"
    assert resumed_run["config"]["resume_storage"]["storage_id"] == "original-volume"
    assert resumed_run["config"]["resume_storage"]["source_resource_id"]
    response = client.post(f"/api/experiments/{resumed_run['id']}/runpod/quote", json={
        "image": "registry/audioenv:v1", "image_digest": "sha256:" + "a" * 64,
        "max_runtime_hours": 1, "max_total_budget_usd": 10,
        "network_volume_id": "different-volume",
    })
    assert response.status_code == 409
    assert "original persisted storage" in response.text

    unverified = app.state.artifacts.put_bytes(
        b"unverified checkpoint", media_type="application/octet-stream",
    )
    repository.add_artifact(
        unverified.id, experiment_id=source.id, project_id=source.project_id,
        kind="checkpoint", media_type=unverified.media_type,
        size_bytes=unverified.size_bytes, path=str(unverified.path),
        metadata={"remote_relative_path": "global_step_5/adapter/model.safetensors"},
    )
    rejected = client.post(f"/api/experiments/{source.id}/resume", json={
        "checkpoint_artifact_id": unverified.id, "run_name": "unsafe resume",
        "additional_steps": 1,
    })
    assert rejected.status_code == 409
    assert "successful persisted checkpoint reload evidence" in rejected.text


def test_api_uses_persisted_records(tmp_path):
    client = TestClient(create_app(f"sqlite:///{tmp_path / 'api.sqlite'}"))
    assert client.get("/api/health").json()["status"] == "ok"
    created = client.post("/api/experiments", json={"name": "api run", "config": {"steps": 2}})
    assert created.status_code == 201
    experiment = created.json()
    listing = client.get("/api/experiments").json()
    assert listing[0]["id"] == experiment["id"]
    invalid = client.post(
        f"/api/experiments/{experiment['id']}/transition", json={"target": "running", "payload": {}}
    )
    assert invalid.status_code == 409


def test_api_rejects_secret_values_and_redacts_legacy_records(tmp_path):
    app = create_app(f"sqlite:///{tmp_path / 'secret-boundary.sqlite'}")
    client = TestClient(app)

    rejected_run = client.post("/api/experiments", json={
        "name": "unsafe", "config": {"provider": {"api_token": "plaintext-secret"}},
    })
    assert rejected_run.status_code == 422
    assert "store an environment/provider secret name" in rejected_run.text
    rejected_model = client.post("/api/models", json={
        "id": "unsafe/model", "display_name": "Unsafe", "adapter_type": "custom",
        "role": "agent", "config": {"password": "plaintext-secret"},
    })
    assert rejected_model.status_code == 422

    accepted = client.post("/api/experiments", json={
        "name": "safe references", "config": {
            "api_key_secret_name": "RUNPOD_API_KEY",
            "secret_names": ["HF_TOKEN"], "token_logprobs": True,
        },
    })
    assert accepted.status_code == 201
    assert accepted.json()["config"]["api_key_secret_name"] == "RUNPOD_API_KEY"

    legacy = app.state.repository.create_experiment(
        "legacy unsafe", {"provider": {"api_token": "legacy-plaintext"}},
    )
    app.state.repository.add_log(
        legacy.id, "legacy event", payload={"password": "legacy-password", "safe": "visible"},
    )
    public = client.get(f"/api/experiments/{legacy.id}").json()
    assert public["config"]["provider"]["api_token"] == "[REDACTED]"
    logs = client.get(f"/api/experiments/{legacy.id}/logs").json()
    assert logs[0]["payload"] == {"password": "[REDACTED]", "safe": "visible"}
    assert "legacy-plaintext" not in json.dumps(client.get("/api/overview").json())


def test_custom_environment_api_persists_validated_boundary(tmp_path):
    client = TestClient(create_app(f"sqlite:///{tmp_path / 'custom.sqlite'}"))
    response = client.post(
        "/api/environments/custom",
        json={
            "environment_id": "custom/ticket",
            "description": "Close a ticket",
            "task_id": "t1",
            "goal": "Close it",
            "initial_state": {"ticket": {"status": "open"}},
            "expected_state": {"ticket": {"status": "closed"}},
            "initial_utterance": "Please close it",
            "tools": [{
                "name": "close_ticket", "description": "Close it", "operation": "set_path",
                "path": ["ticket", "status"], "required_arguments": ["status"], "value_argument": "status",
            }],
            "required_tools": [{"name": "close_ticket", "arguments": {"status": "closed"}}],
            "required_communications": ["closed"],
        },
    )
    assert response.status_code == 201
    assert response.json()["validation"]["passed"] is True
    listing = client.get("/api/environments").json()
    assert listing[0]["authoring_status"] == "validated"


def test_registered_model_stays_unavailable_until_probe_succeeds(tmp_path):
    client = TestClient(create_app(f"sqlite:///{tmp_path / 'models.sqlite'}"))
    model_id = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    registered = client.post("/api/models", json={
        "id": model_id,
        "display_name": "Qwen3 Omni",
        "adapter_type": "qwen3-omni",
        "role": "agent",
        "config": {"checkpoint": str(tmp_path / "missing")},
    })
    assert registered.status_code == 201
    assert registered.json()["available"] is False
    assert registered.json()["registration_state"] == "generated_needs_review"
    probe = client.post("/api/models/probe", json={"model_id": model_id})
    assert probe.status_code == 200
    assert probe.json()["available"] is False
    assert probe.json()["probe"]["reasons"]


def test_dependency_probe_cannot_bypass_model_conformance(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'model-state.sqlite'}")
    database.create_schema()
    repository = PlatformRepository(database)
    repository.register_model("lab/omni", "Omni", "custom", "agent", {})
    pending = repository.set_model_probe("lab/omni", {
        "available": True, "device": "cuda", "reasons": [],
    })
    assert pending.available is False
    assert pending.registration_state == "generated_needs_review"
    accepted = repository.set_model_probe("lab/omni", {
        "available": True, "device": "cuda", "reasons": [],
        "registration_state": "conformant",
    })
    assert accepted.available is True
    assert accepted.registration_state == "conformant"


def test_experiment_export_redacts_nested_secrets(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'export.sqlite'}")
    database.create_schema()
    repository = PlatformRepository(database)
    experiment = repository.create_experiment(
        "redacted run",
        {
            "trainer": "tiny",
            "provider": {"api_token": "do-not-export", "region": "local"},
            "secret_reference": "ENV:MODEL_TOKEN",
        },
    )

    path = export_experiment(repository, experiment.id, tmp_path / "bundle")
    payload = json.loads(path.read_text())

    assert payload["experiment"]["config"]["provider"]["api_token"] == "[REDACTED]"
    assert payload["experiment"]["config"]["secret_reference"] == "[REDACTED]"
    assert payload["integrity"]["config_hash"] == experiment.config_hash
    assert payload["integrity"]["artifact_manifest_sha256"] == sha256(b"[]").hexdigest()
    assert payload["reproducibility"] == {
        "status": "incomplete", "missing": ["experiment_not_completed"],
        "checkpoint_lineage": {
            "parent_experiment_id": None, "checkpoint_artifact_id": None,
        },
    }
    assert "do-not-export" not in path.read_text()


def test_abandoned_worker_marks_job_and_experiment_failed(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'recovery.sqlite'}")
    database.create_schema()
    repository = PlatformRepository(database)
    experiment = repository.create_experiment("recovery run", {"trainer": "tiny"})
    for target in ("validating", "queued", "running"):
        repository.transition(experiment.id, target)
    job = repository.create_job(experiment.id, "training", {})
    claimed = repository.claim_job(job.id, "worker-1")
    assert claimed is not None
    with database.session() as session:
        persisted = session.get(JobRecord, job.id)
        assert persisted is not None
        persisted.heartbeat_at = utcnow() - timedelta(minutes=5)
        session.commit()

    assert repository.recover_abandoned_jobs(stale_after_seconds=60) == 1
    assert repository.get_experiment(experiment.id).status == "failed"
    with database.session() as session:
        assert session.get(JobRecord, job.id).status == "failed"

