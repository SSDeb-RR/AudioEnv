import json

import pytest

from audioenv.training.tau2_dataset import build_tau2_rollout_dataset


class _Discovery:
    def discover(self):
        return {"tasks": [
            {"id": "2", "split": "test"}, {"id": "1", "split": "train"},
            {"id": "0", "split": "train"}, {"id": "ignored", "split": "base"},
        ]}


def test_tau2_dataset_is_immutable_public_and_split_specific(tmp_path):
    output = tmp_path / "train.jsonl"
    manifest = build_tau2_rollout_dataset(
        _Discovery(), output, split="train", experiment_id="exp",
        environment_hash="a" * 64, verifier_hash="b" * 64,
        policy_version="p1", checkpoint_id="base", seeds=(3, 4),
    )
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert manifest.task_ids == ("0", "1") and manifest.row_count == 4
    assert {row["dataset_split"] for row in rows} == {"train"}
    assert {row["task_id"] for row in rows} == {"0", "1"}
    assert {row["seed"] for row in rows} == {3, 4}
    assert all("user_scenario" not in json.dumps(row) for row in rows)
    saved = json.loads((tmp_path / "train.jsonl.manifest.json").read_text())
    assert saved["dataset_sha256"] == manifest.dataset_sha256


def test_tau2_dataset_rejects_missing_split_or_bad_hash(tmp_path):
    with pytest.raises(ValueError, match="lowercase sha256"):
        build_tau2_rollout_dataset(
            _Discovery(), tmp_path / "bad.jsonl", split="train", experiment_id="exp",
            environment_hash="bad", verifier_hash="b" * 64,
            policy_version="p1", checkpoint_id="base",
        )
    with pytest.raises(ValueError, match="no test"):
        build_tau2_rollout_dataset(
            type("Empty", (), {"discover": lambda self: {"tasks": []}})(),
            tmp_path / "empty.jsonl", split="test", experiment_id="exp",
            environment_hash="a" * 64, verifier_hash="b" * 64,
            policy_version="p1", checkpoint_id="base",
        )
