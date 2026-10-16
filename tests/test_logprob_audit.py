from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from audioenv.training.logprob_audit import (
    assess_logprob_parity,
    install_logprob_audit,
    validate_logprob_evidence,
)


def test_logprob_parity_uses_only_trainable_response_tokens():
    report = assess_logprob_parity(
        torch.tensor([[-0.1, 9.0, -0.3]]),
        torch.tensor([[-0.11, -8.0, -0.31]]),
        torch.tensor([[1, 0, 1]]), tolerance=0.02,
    )
    assert report["passed"] and report["token_count"] == 2
    assert report["max_abs_error"] == pytest.approx(0.01)


def test_logprob_parity_rejects_drift_and_nonfinite_values():
    failed = assess_logprob_parity(
        torch.tensor([-0.1]), torch.tensor([-0.4]), torch.tensor([1]), tolerance=0.05,
    )
    assert not failed["passed"]
    with pytest.raises(ValueError, match="finite"):
        assess_logprob_parity(
            torch.tensor([float("nan")]), torch.tensor([-0.4]), torch.tensor([1]), tolerance=0.05,
        )


def test_logprob_evidence_summary_fails_closed(tmp_path: Path):
    good = {
        "schema_version": "audioenv.logprob-parity/v1", "step": 1,
        "token_count": 2, "max_abs_error": 0.01, "mean_abs_error": 0.005,
        "tolerance": 0.05, "passed": True,
    }
    (tmp_path / "step-000001.json").write_text(json.dumps(good), encoding="utf-8")
    assert validate_logprob_evidence(tmp_path)["all_passed"]
    good["passed"] = False
    (tmp_path / "step-000001.json").write_text(json.dumps(good), encoding="utf-8")
    with pytest.raises(ValueError, match="parity failed"):
        validate_logprob_evidence(tmp_path)


def test_logprob_audit_wraps_actual_v1_trainer_boundary(tmp_path: Path, monkeypatch):
    values = {
        "rollout_log_probs": torch.tensor([[-0.1, 0.0, -0.3]]),
        "old_log_probs": torch.tensor([[-0.11, 9.0, -0.31]]),
        "response_mask": torch.tensor([[1, 0, 1]]),
    }

    class Nested:
        def to_padded_tensor(self):
            return values

    class Queue:
        @staticmethod
        def kv_batch_get(*, keys, partition_id, select_fields):
            assert keys == ("episode",) and partition_id == "train"
            assert select_fields == ["rollout_log_probs", "old_log_probs", "response_mask"]
            return Nested()

    class Trainer:
        global_steps = 7

        def _compute_old_log_prob(self, batch, metrics):
            metrics["actor/entropy"] = 0.5
            return batch

    monkeypatch.setenv("AUDIOENV_LOGPROB_EVIDENCE_DIR", str(tmp_path))
    monkeypatch.setenv("AUDIOENV_LOGPROB_MAX_ABS_ERROR", "0.02")
    assert install_logprob_audit(Trainer, queue_module=SimpleNamespace(tq=Queue))
    batch = SimpleNamespace(keys=("episode",), partition_id="train")
    metrics = {}
    assert Trainer()._compute_old_log_prob(batch, metrics) is batch
    assert metrics["audioenv/logprob_max_abs_error"] == pytest.approx(0.01)
    summary = validate_logprob_evidence(tmp_path)
    assert summary["all_passed"] and summary["steps"] == 1
