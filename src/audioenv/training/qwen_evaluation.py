"""Protocol-bound evaluation records from VeRL validation dumps."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable

from ..contracts import EvaluationProtocolV1, EvaluationResultV1, EvaluationSampleV1
from ..evaluation import build_evaluation_result, compare_evaluations


def evaluation_protocol_from_dataset(
    dataset_path: Path,
    *,
    checkpoint_id: str,
    caller_revision: str,
    caller_config: dict[str, Any],
    audio_pipeline: dict[str, Any],
    sampling: dict[str, Any],
) -> EvaluationProtocolV1:
    manifest_path = dataset_path.with_suffix(dataset_path.suffix + ".manifest.json")
    manifest_payload = manifest_path.read_bytes()
    rows = _jsonl(dataset_path)
    if not rows:
        raise ValueError("held-out dataset is empty")
    environment_hashes = {row.get("environment_hash") for row in rows}
    verifier_hashes = {row.get("verifier_hash") for row in rows}
    splits = {row.get("dataset_split") for row in rows}
    if len(environment_hashes) != 1 or len(verifier_hashes) != 1 or splits != {"test"}:
        raise ValueError("held-out dataset protocol metadata is inconsistent")
    return EvaluationProtocolV1(
        environment_hash=str(environment_hashes.pop()),
        verifier_hash=str(verifier_hashes.pop()),
        split_manifest_hash=sha256(manifest_payload).hexdigest(), split="test",
        caller_hash=_hash({"revision": caller_revision, "config": caller_config}),
        audio_pipeline_hash=_hash(audio_pipeline), sampling_hash=_hash(sampling),
        checkpoint_id=checkpoint_id,
        seeds=tuple(sorted({int(row.get("seed", 0)) for row in rows})),
    )


def evaluation_from_verl_dump(
    dump: Path, protocol: EvaluationProtocolV1,
    *, expected_keys: set[tuple[str, str, int]] | None = None,
) -> EvaluationResultV1:
    rows = _jsonl(dump)
    samples = []
    for row in rows:
        task_id = row.get("task_id")
        seed = row.get("seed")
        profile = row.get("audio_profile_id", "native-clean")
        if not isinstance(task_id, str) or not task_id or not isinstance(seed, int):
            raise ValueError("VeRL evaluation dump lacks task_id or seed reward evidence")
        if row.get("verifier_hash") != protocol.verifier_hash:
            raise ValueError("VeRL evaluation reward evidence has a verifier hash mismatch")
        score = float(row.get("score", 0.0))
        samples.append(EvaluationSampleV1(
            task_id=task_id, audio_profile_id=str(profile), seed=seed,
            passed=bool(float(row.get("terminal_reward", score)) > 0.0), reward=score,
            safety_violations=int(row.get("safety_violations", 0)),
            invalid_calls=int(row.get("invalid_calls", 0)),
            repeated_failed_calls=int(row.get("repeated_failed_calls", 0)),
            failure_category=str(row["failure_category"]) if row.get("failure_category") else None,
        ))
    keys = {(sample.task_id, sample.audio_profile_id, sample.seed) for sample in samples}
    if len(keys) != len(samples):
        raise ValueError("VeRL evaluation dump contains duplicate sample identities")
    if expected_keys is not None and keys != expected_keys:
        raise ValueError("VeRL evaluation dump does not match the immutable held-out samples")
    return build_evaluation_result(protocol, samples)


def evaluation_sample_keys_from_dataset(dataset_path: Path) -> set[tuple[str, str, int]]:
    return {
        (str(row["task_id"]), str(row.get("audio_profile_id", "native-clean")), int(row.get("seed", 0)))
        for row in _jsonl(dataset_path)
    }


def verify_reloaded_checkpoint(
    before_reload: EvaluationResultV1, after_reload: EvaluationResultV1,
) -> tuple[bool, tuple[str, ...]]:
    comparison = compare_evaluations(before_reload, after_reload)
    reasons: list[str] = []
    if not comparison.compatible:
        reasons.extend(comparison.incompatibilities)
    before = _sample_signature(before_reload.samples)
    after = _sample_signature(after_reload.samples)
    if before != after:
        reasons.append("sample outcomes differ after checkpoint reload")
    return not reasons, tuple(reasons)


def write_evaluation_bundle(
    output: Path, baseline: EvaluationResultV1, trained: EvaluationResultV1,
    *, reload_verified: bool, reload_reasons: Iterable[str], checkpoint_sha256: str,
) -> dict[str, Path]:
    comparison = compare_evaluations(baseline, trained)
    if not comparison.compatible:
        raise ValueError("baseline and trained evaluation protocols are incompatible: " + ", ".join(comparison.incompatibilities))
    output.mkdir(parents=True, exist_ok=True)
    payloads = {
        "baseline": baseline.model_dump(mode="json"),
        "trained": trained.model_dump(mode="json"),
        "comparison": comparison.model_dump(mode="json"),
        "checkpoint-reload": {
            "schema_version": "audioenv.checkpoint-reload/v1",
            "verified": reload_verified, "reasons": list(reload_reasons),
            "checkpoint_sha256": checkpoint_sha256,
            "trained_protocol_hash": trained.protocol.protocol_hash,
        },
    }
    paths = {}
    for name, payload in payloads.items():
        path = output / f"{name}.json"
        path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        paths[name] = path
    return paths


def newest_dump(directory: Path, *, exclude_step: int | None = None) -> Path:
    candidates = []
    for path in directory.glob("*.jsonl"):
        try:
            step = int(path.stem)
        except ValueError:
            continue
        if exclude_step is None or step != exclude_step:
            candidates.append((step, path))
    if not candidates:
        raise FileNotFoundError(f"no VeRL validation dumps found in {directory}")
    return max(candidates)[1]


def _jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSONL at {path}: {exc}") from exc
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"JSONL at {path} must contain objects")
    return rows


def _sample_signature(samples: Iterable[EvaluationSampleV1]) -> tuple[tuple[Any, ...], ...]:
    return tuple(sorted(
        (row.task_id, row.audio_profile_id, row.seed, row.passed, row.reward,
         row.safety_violations, row.invalid_calls, row.repeated_failed_calls, row.failure_category)
        for row in samples
    ))


def _hash(payload: Any) -> str:
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
