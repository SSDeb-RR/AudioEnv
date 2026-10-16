from __future__ import annotations

import json
from pathlib import Path
import pytest

from audioenv.training.qwen_evaluation import (
    evaluation_from_verl_dump, evaluation_protocol_from_dataset,
    verify_reloaded_checkpoint, write_evaluation_bundle,
)
from audioenv.training.tau2_dataset import build_tau2_rollout_dataset


class _Discovery:
    def discover(self):
        return {"tasks": [{"id": "held-1", "split": "test"}, {"id": "held-2", "split": "test"}]}


def _fixture(tmp_path: Path):
    dataset = tmp_path / "test.jsonl"
    build_tau2_rollout_dataset(
        _Discovery(), dataset, split="test", experiment_id="exp",
        environment_hash="a" * 64, verifier_hash="b" * 64,
        policy_version="p1", checkpoint_id="base", seeds=(11,),
    )
    protocol = evaluation_protocol_from_dataset(
        dataset, checkpoint_id="base", caller_revision="caller-rev",
        caller_config={"temperature": 0.4}, audio_pipeline={"profile": "native-clean"},
        sampling={"temperature": 0.7},
    )
    dump = tmp_path / "0.jsonl"
    rows = [
        {"task_id": "held-1", "seed": 11, "audio_profile_id": "native-clean", "score": 0.0, "terminal_reward": 0.0, "verifier_hash": "b" * 64},
        {"task_id": "held-2", "seed": 11, "audio_profile_id": "native-clean", "score": 1.0, "terminal_reward": 1.0, "verifier_hash": "b" * 64},
    ]
    dump.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return protocol, dump


def test_qwen_evaluation_builds_protocol_bound_results_and_export(tmp_path: Path):
    protocol, dump = _fixture(tmp_path)
    baseline = evaluation_from_verl_dump(dump, protocol)
    trained_protocol = protocol.model_copy(update={"checkpoint_id": "checkpoint-hash"})
    trained = evaluation_from_verl_dump(dump, trained_protocol)
    verified, reasons = verify_reloaded_checkpoint(trained, trained)
    assert verified and not reasons
    paths = write_evaluation_bundle(
        tmp_path / "evaluations", baseline, trained, reload_verified=True,
        reload_reasons=(), checkpoint_sha256="c" * 64,
    )
    comparison = json.loads(paths["comparison"].read_text())
    assert comparison["compatible"] and comparison["paired_sample_count"] == 2
    assert baseline.protocol.compatibility_hash == trained.protocol.compatibility_hash


def test_checkpoint_reload_requires_identical_sample_outcomes(tmp_path: Path):
    protocol, dump = _fixture(tmp_path)
    before = evaluation_from_verl_dump(dump, protocol)
    rows = [json.loads(line) for line in dump.read_text().splitlines()]
    rows[0]["score"] = 1.0; rows[0]["terminal_reward"] = 1.0
    dump.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    after = evaluation_from_verl_dump(dump, protocol)
    verified, reasons = verify_reloaded_checkpoint(before, after)
    assert not verified and "sample outcomes differ after checkpoint reload" in reasons


def test_qwen_evaluation_rejects_missing_reward_identity(tmp_path: Path):
    protocol, dump = _fixture(tmp_path)
    dump.write_text('{"score":1}\n', encoding="utf-8")
    try:
        evaluation_from_verl_dump(dump, protocol)
    except ValueError as exc:
        assert "lacks task_id or seed" in str(exc)
    else:
        raise AssertionError("missing evaluation identity was accepted")


def test_tau2_dataset_refuses_unimplemented_audio_profile(tmp_path: Path):
    with pytest.raises(ValueError, match="only the implemented native-clean"):
        build_tau2_rollout_dataset(
            _Discovery(), tmp_path / "test.jsonl", split="test", experiment_id="exp",
            environment_hash="a" * 64, verifier_hash="b" * 64,
            policy_version="p1", checkpoint_id="base", audio_profile_id="fake-accent",
        )
