from datetime import timedelta
import json

import pytest
from fastapi.testclient import TestClient

from audioenv.platform.api import create_app
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

