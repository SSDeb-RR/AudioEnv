import json
from pathlib import Path
import sys

import pytest

from audioenv.integrations.tau2_worker import RetailWorker, _reference_summary
from audioenv.integrations.tau2_client import (
    Tau2WorkerClient,
    Tau2WorkerError,
    Tau2WorkerTimeout,
)


ECHO_WORKER = r"""
import json, sys
for line in sys.stdin:
    request = json.loads(line)
    operation = request["operation"]
    if operation == "fail":
        response = {"protocol": request["protocol"], "request_id": request["request_id"], "ok": False, "error": "rejected"}
    else:
        response = {"protocol": request["protocol"], "request_id": request["request_id"], "ok": True, "result": {"operation": operation, "payload": request.get("arguments", {})}}
    print(json.dumps(response), flush=True)
"""


def test_tau2_client_correlates_stateful_jsonl_requests_and_closes():
    client = Tau2WorkerClient((sys.executable, "-u", "-c", ECHO_WORKER), timeout_seconds=2)
    with client:
        assert client.probe()["operation"] == "probe"
        result = client.tool("session", "lookup", {"order_id": "O-1"})
        assert result["payload"] == {"order_id": "O-1"}
    assert client._process is None


def test_tau2_client_surfaces_worker_errors():
    with Tau2WorkerClient((sys.executable, "-u", "-c", ECHO_WORKER), timeout_seconds=2) as client:
        with pytest.raises(Tau2WorkerError, match="rejected"):
            client.request("fail")


def test_tau2_client_times_out_and_terminates_process():
    script = "import time; time.sleep(60)"
    client = Tau2WorkerClient((sys.executable, "-u", "-c", script), timeout_seconds=0.05)
    with pytest.raises(Tau2WorkerTimeout, match="timed out"):
        client.probe()
    assert client._process is None


def test_tau2_client_rejects_uncorrelated_response():
    script = r'''import json, sys
for line in sys.stdin:
 r=json.loads(line); print(json.dumps({"protocol":r["protocol"],"request_id":"wrong","ok":True,"result":{}}), flush=True)
'''
    client = Tau2WorkerClient((sys.executable, "-u", "-c", script), timeout_seconds=2)
    with pytest.raises(Tau2WorkerError, match="correlation mismatch"):
        client.probe()
    assert client._process is None


def test_tau2_discovery_does_not_replace_official_splits_with_base():
    worker = object.__new__(RetailWorker)
    worker.splits = {"train": ["1"], "test": ["2"], "base": ["1", "2"]}
    worker.tasks = {
        "1": type("Task", (), {"id": "1"})(),
        "2": type("Task", (), {"id": "2"})(),
    }
    discovered = worker.handle({"operation": "discover"})
    assert discovered == {"tasks": [{"id": "1", "split": "train"}, {"id": "2", "split": "test"}]}


def test_tau2_private_caller_context_is_separate_from_public_reset():
    worker = object.__new__(RetailWorker)
    scenario = type("Scenario", (), {"model_dump": lambda self, mode=None: {"instructions": {"reason_for_call": "refund"}}})()
    task = type("Task", (), {"id": "1", "user_scenario": scenario})()
    worker.sessions = {"session": {"task": task}}
    assert worker.handle({"operation": "caller_context", "session_id": "session"}) == {
        "user_scenario": {"instructions": {"reason_for_call": "refund"}},
    }


def test_tau2_reference_validation_is_disabled_in_rollout_workers(monkeypatch):
    monkeypatch.delenv("AUDIOENV_TAU2_VALIDATION_MODE", raising=False)
    worker = object.__new__(RetailWorker)
    with pytest.raises(PermissionError, match="disabled in rollout workers"):
        worker.handle({"operation": "validate_reference", "task_id": "0"})


def test_tau2_reference_evidence_redacts_protected_trajectory_details():
    summary = _reference_summary("0", {
        "upstream_reward": 1.0, "upstream_environment_reward": 1.0,
        "db_match": True, "user_db_match": True,
        "reward_info": {
            "reward": 1.0, "reward_basis": ["DB"], "reward_breakdown": {"DB": 1.0},
            "db_check": {"db_match": True},
            "action_checks": [{"action": {"name": "protected_tool"}}],
            "communicate_checks": [{"criteria": "protected text"}],
        },
    })
    serialized = json.dumps(summary)
    assert summary["upstream_reward"] == 1.0
    assert "protected_tool" not in serialized
    assert "protected text" not in serialized
