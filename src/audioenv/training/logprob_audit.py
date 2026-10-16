"""Fail-closed rollout/training log-probability parity evidence."""

from __future__ import annotations

from functools import wraps
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from typing import Any


def assess_logprob_parity(rollout, recomputed, response_mask, *, tolerance: float) -> dict[str, Any]:
    import torch
    if rollout.shape != recomputed.shape or rollout.shape != response_mask.shape:
        raise ValueError("rollout, recomputed, and response-mask shapes must match")
    selected = response_mask.bool()
    if not bool(selected.any()):
        raise ValueError("log-probability parity requires trainable response tokens")
    left, right = rollout[selected].float(), recomputed[selected].float()
    if not bool(torch.isfinite(left).all() and torch.isfinite(right).all()):
        raise ValueError("log-probability parity inputs must be finite")
    difference = (left - right).abs()
    maximum = float(difference.max().item())
    mean = float(difference.mean().item())
    return {
        "token_count": int(difference.numel()), "max_abs_error": maximum,
        "mean_abs_error": mean, "tolerance": tolerance, "passed": maximum <= tolerance,
    }


def install_logprob_audit(trainer_class=None, *, queue_module=None) -> bool:
    if trainer_class is None:
        from verl.trainer.ppo.v1.trainer_base import PPOTrainer
        trainer_class = PPOTrainer
    if getattr(trainer_class, "_audioenv_logprob_audit", False):
        return False
    original = trainer_class._compute_old_log_prob
    if queue_module is None:
        from verl.trainer.ppo.v1 import trainer_base as queue_module

    @wraps(original)
    def compute_old_log_prob(self, batch, metrics):
        result = original(self, batch, metrics)
        fields = ["rollout_log_probs", "old_log_probs", "response_mask"]
        nested = queue_module.tq.kv_batch_get(
            keys=batch.keys, partition_id=batch.partition_id, select_fields=fields,
        )
        data = nested.to_padded_tensor()
        tolerance = float(os.environ.get("AUDIOENV_LOGPROB_MAX_ABS_ERROR", "0.05"))
        evidence = assess_logprob_parity(
            data["rollout_log_probs"], data["old_log_probs"], data["response_mask"],
            tolerance=tolerance,
        )
        evidence.update({
            "schema_version": "audioenv.logprob-parity/v1",
            "step": int(getattr(self, "global_steps", 0)),
        })
        _write(evidence)
        metrics["audioenv/logprob_max_abs_error"] = evidence["max_abs_error"]
        metrics["audioenv/logprob_mean_abs_error"] = evidence["mean_abs_error"]
        if not evidence["passed"]:
            raise RuntimeError(
                f"rollout/training log-probability drift {evidence['max_abs_error']} "
                f"exceeds {tolerance}"
            )
        return result

    trainer_class._compute_old_log_prob = compute_old_log_prob
    trainer_class._audioenv_logprob_audit = True
    return True


def validate_logprob_evidence(directory: Path) -> dict[str, Any]:
    paths = sorted(directory.glob("step-*.json"))
    if not paths:
        raise ValueError("no rollout/training log-probability evidence was produced")
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if any(row.get("schema_version") != "audioenv.logprob-parity/v1" for row in rows):
        raise ValueError("unsupported log-probability evidence")
    if any(not row.get("passed") or not math.isfinite(float(row.get("max_abs_error", math.inf))) for row in rows):
        raise ValueError("rollout/training log-probability parity failed")
    return {
        "schema_version": "audioenv.logprob-summary/v1", "steps": len(rows),
        "max_abs_error": max(float(row["max_abs_error"]) for row in rows),
        "mean_abs_error": sum(float(row["mean_abs_error"]) for row in rows) / len(rows),
        "all_passed": True,
        "summary_sha256": sha256(b"".join(path.read_bytes() for path in paths)).hexdigest(),
    }


def _write(payload: dict[str, Any]) -> None:
    root_value = os.environ.get("AUDIOENV_LOGPROB_EVIDENCE_DIR")
    if not root_value:
        raise RuntimeError("AUDIOENV_LOGPROB_EVIDENCE_DIR is required")
    root = Path(root_value); root.mkdir(parents=True, exist_ok=True)
    step = int(payload["step"])
    destination = root / f"step-{step:06d}.json"
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
