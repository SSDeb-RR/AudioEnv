import json

import pytest
from fastapi.testclient import TestClient

from audioenv.platform.api import create_app
from audioenv.promotion import PromotionEvidenceAuthority, aggregate_validation_checks


def _validation():
    names = {
        "tool_handlers_complete": True,
        "audio_profile:clean": True,
        "task:oracle_pass": True,
        "task:noop_fail": True,
        "task:reward_mutation": True,
        "task:deterministic_replay": True,
        "task:state_isolation": True,
        "task:no_text_leak": True,
    }
    return {"passed": True, "checks": [{"name": name, "passed": passed} for name, passed in names.items()]}


def _custom_payload():
    return {
        "environment_id": "custom/promotion-test",
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
    }


def test_backend_evidence_is_signed_and_tampering_fails(tmp_path):
    authority = PromotionEvidenceAuthority(tmp_path / "key")
    digest = "a" * 64
    evidence = authority.issue(
        environment_hash=digest, verifier_hash_before=digest,
        verifier_hash_after=digest, validation=_validation(),
    )
    assert authority.verify(evidence).promoted is True
    tampered = evidence.model_copy(update={"checks": {**evidence.checks, "oracle_pass": False}})
    with pytest.raises(ValueError, match="signature is invalid"):
        authority.verify(tampered)
    assert (tmp_path / "key").stat().st_mode & 0o777 == 0o600


def test_missing_backend_checks_cannot_be_promoted():
    checks = aggregate_validation_checks({"passed": True, "checks": []})
    assert checks and not any(checks.values())


def test_api_promotes_only_from_backend_issued_artifact(tmp_path):
    app = create_app(f"sqlite:///{tmp_path / 'api.sqlite'}")
    client = TestClient(app)
    created = client.post("/api/environments/custom", json=_custom_payload())
    assert created.status_code == 201
    environment_id = created.json()["id"]
    environment_hash = created.json()["package"]["content_hash"]

    # The former client-supplied booleans are rejected by the strict request schema.
    forged = client.post(f"/api/environments/{environment_id}/promote", json={
        "verifier_hash_after": "f" * 64,
        "checks": {"oracle_pass": True},
    })
    assert forged.status_code == 422

    issued = client.post(f"/api/environments/{environment_id}/promotion-evidence")
    assert issued.status_code == 201
    evidence = issued.json()
    assert evidence["report"]["promoted"] is True
    promoted = client.post(f"/api/environments/{environment_id}/promote", json={
        "evidence_artifact_id": evidence["artifact_id"],
        "confirmation_hash": environment_hash,
    })
    assert promoted.status_code == 200
    assert promoted.json()["authoring_status"] == "validated"
    assert promoted.json()["validation"]["evidence_artifact_id"] == evidence["artifact_id"]

    artifact = app.state.repository.get_artifact(evidence["artifact_id"])
    assert artifact is not None and artifact.kind == "promotion-evidence"
    assert json.loads(artifact.metadata_json)["environment_id"] == environment_id
