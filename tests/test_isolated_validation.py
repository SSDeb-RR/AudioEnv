import json
import subprocess
import sys

import pytest

from audioenv.authoring import CustomTaskInput
from audioenv.isolated_validation import IsolatedValidationRunner
from audioenv.validation_worker import MAX_REQUEST_BYTES, PROTOCOL_VERSION


def _task():
    return CustomTaskInput.model_validate({
        "environment_id": "custom/isolated",
        "description": "Update status",
        "task_id": "t1",
        "goal": "Close it",
        "initial_state": {"ticket": {"status": "open"}},
        "expected_state": {"ticket": {"status": "closed"}},
        "initial_utterance": "Close the ticket",
        "tools": [{
            "name": "close", "description": "close", "operation": "set_path",
            "path": ["ticket", "status"], "required_arguments": ["status"], "value_argument": "status",
        }],
        "required_tools": [{"name": "close", "arguments": {"status": "closed"}}],
        "required_communications": ["closed"],
    })


def test_isolated_worker_reexecutes_environment_and_returns_hashes():
    first = IsolatedValidationRunner().validate_custom(_task())
    second = IsolatedValidationRunner().validate_custom(_task())
    assert first["passed"] is True
    assert first["worker"] == "audioenv-declarative-validator-v1"
    assert first["environment_hash"] == second["environment_hash"]
    assert first["verifier_hash"] == second["verifier_hash"]
    assert any(row["name"].endswith(":state_isolation") for row in first["checks"])


def test_validation_worker_rejects_oversized_input_without_traceback():
    process = subprocess.run(
        (sys.executable, "-m", "audioenv.validation_worker"),
        input="x" * (MAX_REQUEST_BYTES + 1), capture_output=True, text=True, check=False,
    )
    response = json.loads(process.stdout)
    assert response["protocol"] == PROTOCOL_VERSION
    assert response["ok"] is False
    assert "size limit" in response["error"]
    assert "Traceback" not in process.stderr


def test_isolated_runner_validates_resource_configuration():
    with pytest.raises(ValueError, match="resource limits"):
        IsolatedValidationRunner(timeout_seconds=0)
