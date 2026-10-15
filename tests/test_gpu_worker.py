from __future__ import annotations

import json
from pathlib import Path

import pytest

from audioenv.contracts import GPUWorkerExperimentV1
from audioenv.platform.gpu_worker import load_experiment, run_experiment


def test_worker_bundle_rejects_incompatible_backend():
    with pytest.raises(ValueError, match="local REINFORCE"):
        GPUWorkerExperimentV1(
            experiment_id="bad", run_name="bad", backend="tiny-cpu-proof",
            trainer="qwen3-omni-thinker-gspo-lora", algorithm="gspo", output_dir="out",
        )


def test_provider_neutral_worker_runs_training_and_emits_artifacts(tmp_path: Path):
    bundle = GPUWorkerExperimentV1(
        experiment_id="worker-proof", run_name="worker-proof", backend="tiny-cpu-proof",
        trainer="audioenv/small-audio-reinforce", algorithm="reinforce",
        output_dir=str(tmp_path / "output"),
        hyperparameters={"steps": 2, "checkpoint_interval": 1, "seed": 9},
        dependency_pins={"audioenv": "test"},
    )
    path = tmp_path / "experiment.json"
    path.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")
    assert load_experiment(path).content_hash == bundle.content_hash
    assert run_experiment(bundle) == 0
    lines = [json.loads(line) for line in (tmp_path / "output" / "events.jsonl").read_text().splitlines()]
    event_types = [line["type"] for line in lines]
    assert event_types[0] == "worker.started"
    assert "preflight.completed" in event_types
    assert "training.started" in event_types
    assert "checkpoint.saved" in event_types
    assert "evaluation.completed" in event_types
    assert event_types[-1] == "worker.completed"
    manifest = json.loads((tmp_path / "output" / "worker-artifacts.json").read_text())
    assert any(row["path"] == "final.pt" for row in manifest)


def test_worker_cancellation_is_a_terminal_event(tmp_path: Path):
    cancel = tmp_path / "cancel"
    cancel.write_text("stop", encoding="utf-8")
    bundle = GPUWorkerExperimentV1(
        experiment_id="cancel-proof", run_name="cancel-proof", backend="tiny-cpu-proof",
        trainer="audioenv/small-audio-reinforce", algorithm="reinforce",
        output_dir=str(tmp_path / "cancelled"), cancellation_file=str(cancel),
        hyperparameters={"steps": 5, "checkpoint_interval": 1},
    )
    assert run_experiment(bundle) == 130
    events = (tmp_path / "cancelled" / "events.jsonl").read_text()
    assert '"type":"worker.cancelled"' in events
