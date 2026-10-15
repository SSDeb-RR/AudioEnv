"""Leakage-aware reproducibility export for a persisted experiment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .db import PlatformRepository, serialize_experiment


SENSITIVE_FRAGMENTS = ("secret", "password", "token", "api_key", "credential")


def export_experiment(repository: PlatformRepository, experiment_id: str, destination: Path) -> Path:
    experiment = repository.get_experiment(experiment_id)
    if experiment is None:
        raise KeyError(experiment_id)
    destination.mkdir(parents=True, exist_ok=True)
    record = serialize_experiment(experiment)
    record["config"] = _redact(record["config"])
    payload = {
        "schema_version": "audioenv.experiment-export/v1",
        "experiment": record,
        "metrics": [
            {
                "step": row.step,
                "name": row.name,
                "value": row.value,
                "source": row.source,
                "created_at": row.created_at.isoformat(),
            }
            for row in repository.list_metrics(experiment_id)
        ],
        "evaluations": [
            {
                "id": row.id,
                "checkpoint_id": row.checkpoint_id,
                "environment_hash": row.environment_hash,
                "verifier_hash": row.verifier_hash,
                "split": row.split,
                "protocol_hash": row.protocol_hash,
                "config": _redact(json.loads(row.config_json)),
                "metrics": json.loads(row.metrics_json),
                "samples": [
                    {
                        "task_id": sample.task_id,
                        "audio_profile_id": sample.audio_profile_id,
                        "seed": sample.seed,
                        "passed": sample.passed,
                        "reward": sample.reward,
                        **json.loads(sample.details_json),
                    }
                    for sample in repository.list_evaluation_samples(row.id)
                ],
            }
            for row in repository.list_evaluations()
            if row.experiment_id == experiment_id
        ],
        "comparisons": [
            {
                "id": row.id,
                "baseline_evaluation_id": row.baseline_evaluation_id,
                "trained_evaluation_id": row.trained_evaluation_id,
                "compatibility_hash": row.compatibility_hash,
                "result": json.loads(row.result_json),
            }
            for row in repository.list_comparisons(experiment_id)
        ],
        "logs": [
            {
                "level": row.level,
                "message": row.message,
                "payload": _redact(json.loads(row.payload_json)),
                "created_at": row.created_at.isoformat(),
            }
            for row in repository.list_logs(experiment_id)
        ],
        "artifacts": [
            {
                "id": row.id,
                "kind": row.kind,
                "media_type": row.media_type,
                "size_bytes": row.size_bytes,
                "metadata": _redact(json.loads(row.metadata_json)),
            }
            for row in repository.list_artifacts()
            if row.experiment_id == experiment_id
        ],
    }
    path = destination / "experiment.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if any(fragment in key.casefold() for fragment in SENSITIVE_FRAGMENTS) else _redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value

