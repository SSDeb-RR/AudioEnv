"""Compile public tau2 task identifiers into immutable VeRL JSONL inputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Literal, Protocol


class DiscoveryPort(Protocol):
    def discover(self) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class Tau2DatasetManifest:
    split: str
    task_ids: tuple[str, ...]
    row_count: int
    dataset_sha256: str
    configuration_sha256: str
    output_path: str


def build_tau2_rollout_dataset(
    client: DiscoveryPort,
    output: Path,
    *,
    split: Literal["train", "test"],
    experiment_id: str,
    environment_hash: str,
    verifier_hash: str,
    policy_version: str,
    checkpoint_id: str,
    model_id: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    reward_version: str = "tau2-official-v1",
    seeds: tuple[int, ...] = (0,),
    max_turns: int = 16,
    audio_profile_id: str = "native-clean",
    limit: int | None = None,
) -> Tau2DatasetManifest:
    """Write canonical JSONL containing no caller goal or verifier truth."""

    if audio_profile_id != "native-clean":
        raise ValueError("tau2 Qwen datasets currently support only the implemented native-clean audio pipeline")

    for name, value in (("environment_hash", environment_hash), ("verifier_hash", verifier_hash)):
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"{name} must be a lowercase sha256")
    if not seeds or max_turns < 1:
        raise ValueError("at least one seed and a positive max_turns are required")
    discovered = client.discover().get("tasks")
    if not isinstance(discovered, list):
        raise ValueError("tau2 discovery did not return tasks")
    task_ids = tuple(sorted(
        str(item["id"]) for item in discovered
        if isinstance(item, dict) and item.get("split") == split and item.get("id") is not None
    ))
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        task_ids = task_ids[:limit]
    if not task_ids:
        raise ValueError(f"tau2 discovery returned no {split} tasks")

    configuration = {
        "schema_version": "audioenv.tau2-verl-dataset/v1",
        "split": split, "task_ids": task_ids, "seeds": seeds,
        "experiment_id": experiment_id, "environment_hash": environment_hash,
        "verifier_hash": verifier_hash, "policy_version": policy_version,
        "checkpoint_id": checkpoint_id, "model_id": model_id,
        "reward_version": reward_version, "max_turns": max_turns,
        "audio_profile_id": audio_profile_id,
    }
    configuration_hash = _hash(configuration)
    rows = []
    for task_id in task_ids:
        for seed in seeds:
            rows.append({
                "data_source": "audioenv/tau2-retail",
                "prompt": [{"role": "user", "content": "Begin the protected audio task."}],
                "reward_model": {"style": "rule", "ground_truth": "protected"},
                "extra_info": {
                    "audioenv_verifier_hash": verifier_hash,
                    "audioenv_dataset_hash": configuration_hash,
                },
                "task_id": task_id, "seed": seed, "experiment_id": experiment_id,
                "environment_hash": environment_hash, "verifier_hash": verifier_hash,
                "policy_version": policy_version, "checkpoint_id": checkpoint_id,
                "model_id": model_id, "reward_version": reward_version,
                "max_turns": max_turns, "dataset_split": split,
                "audio_profile_id": audio_profile_id,
                "dataset_configuration_hash": configuration_hash,
            })
    payload = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in rows
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    manifest = Tau2DatasetManifest(
        split=split, task_ids=task_ids, row_count=len(rows),
        dataset_sha256=sha256(payload).hexdigest(), configuration_sha256=configuration_hash,
        output_path=str(output.resolve()),
    )
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps(asdict(manifest), sort_keys=True, indent=2) + "\n", encoding="utf-8",
    )
    return manifest


def _hash(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
