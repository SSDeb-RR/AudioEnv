"""Held-out evaluation records and deterministic confidence intervals."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import random
from typing import Iterable

from .adapter import EnvironmentPackage
from .spec import public_dataclass
from .contracts import (
    ComparisonV1,
    EvaluationProtocolV1,
    EvaluationResultV1,
    EvaluationSampleV1,
)


@dataclass(frozen=True, slots=True)
class EvaluationSample:
    task_id: str
    seed: int
    passed: bool
    reward: float
    safety_violations: int = 0
    invalid_calls: int = 0
    repeated_failed_calls: int = 0


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    sample_count: int
    success_rate: float
    success_ci_low: float
    success_ci_high: float
    mean_reward: float
    safety_violations: int
    invalid_calls: int
    repeated_failed_calls: int


def summarize(samples: Iterable[EvaluationSample], *, seed: int = 13, bootstrap_rounds: int = 2_000) -> EvaluationSummary:
    rows = tuple(samples)
    if not rows:
        raise ValueError("evaluation requires at least one sample")
    values = [float(row.passed) for row in rows]
    rng = random.Random(seed)
    estimates = []
    for _ in range(bootstrap_rounds):
        estimates.append(sum(rng.choice(values) for _ in values) / len(values))
    estimates.sort()
    low = estimates[int(0.025 * (len(estimates) - 1))]
    high = estimates[int(0.975 * (len(estimates) - 1))]
    return EvaluationSummary(
        sample_count=len(rows),
        success_rate=sum(values) / len(values),
        success_ci_low=low,
        success_ci_high=high,
        mean_reward=sum(row.reward for row in rows) / len(rows),
        safety_violations=sum(row.safety_violations for row in rows),
        invalid_calls=sum(row.invalid_calls for row in rows),
        repeated_failed_calls=sum(row.repeated_failed_calls for row in rows),
    )


def package_hashes(package: EnvironmentPackage) -> tuple[str, str]:
    public = {
        "spec": asdict(package.spec),
        "tasks": [
            {
                "id": task.id,
                "split": task.split,
                "goal": task.user_goal,
                "initial_state": task.initial_state,
                "initial_utterance": task.initial_utterance,
            }
            for task in package.tasks
        ],
        "public_action_candidates": {
            task_id: [public_dataclass(action) for action in actions]
            for task_id, actions in sorted(package.public_action_candidates.items())
        },
    }
    protected = {
        task.id: {
            "hidden_facts": task.hidden_facts,
            "expected_state": task.expected_state,
            "required_tools": [asdict(item) for item in task.required_tools],
            "required_communications": task.required_communications,
            "forbidden_tools": task.forbidden_tools,
        }
        for task in package.tasks
    }
    return _hash(public), _hash(protected)


def build_evaluation_result(
    protocol: EvaluationProtocolV1,
    samples: Iterable[EvaluationSampleV1],
) -> EvaluationResultV1:
    rows = tuple(samples)
    summary = summarize(
        EvaluationSample(
            task_id=row.task_id,
            seed=row.seed,
            passed=row.passed,
            reward=row.reward,
            safety_violations=row.safety_violations,
            invalid_calls=row.invalid_calls,
            repeated_failed_calls=row.repeated_failed_calls,
        )
        for row in rows
    )
    failures: dict[str, int] = {}
    for row in rows:
        if row.failure_category:
            failures[row.failure_category] = failures.get(row.failure_category, 0) + 1
    return EvaluationResultV1(
        protocol=protocol,
        samples=rows,
        success_rate=summary.success_rate,
        success_ci_low=summary.success_ci_low,
        success_ci_high=summary.success_ci_high,
        mean_reward=summary.mean_reward,
        reward_components={"terminal_success": summary.success_rate},
        failure_counts=failures,
    )


def compare_evaluations(
    baseline: EvaluationResultV1,
    trained: EvaluationResultV1,
    *,
    seed: int = 29,
    bootstrap_rounds: int = 2_000,
) -> ComparisonV1:
    fields = (
        "environment_hash",
        "verifier_hash",
        "split_manifest_hash",
        "split",
        "caller_hash",
        "audio_pipeline_hash",
        "sampling_hash",
        "seeds",
    )
    mismatches = tuple(
        field
        for field in fields
        if getattr(baseline.protocol, field) != getattr(trained.protocol, field)
    )
    compatibility_hash = baseline.protocol.compatibility_hash
    if mismatches:
        return ComparisonV1(
            baseline_protocol_hash=baseline.protocol.protocol_hash,
            trained_protocol_hash=trained.protocol.protocol_hash,
            compatibility_hash=compatibility_hash,
            compatible=False,
            incompatibilities=mismatches,
            paired_sample_count=0,
        )
    baseline_rows = {
        (row.task_id, row.audio_profile_id, row.seed): row for row in baseline.samples
    }
    trained_rows = {
        (row.task_id, row.audio_profile_id, row.seed): row for row in trained.samples
    }
    if baseline_rows.keys() != trained_rows.keys():
        return ComparisonV1(
            baseline_protocol_hash=baseline.protocol.protocol_hash,
            trained_protocol_hash=trained.protocol.protocol_hash,
            compatibility_hash=compatibility_hash,
            compatible=False,
            incompatibilities=("sample_keys",),
            paired_sample_count=0,
        )
    deltas = [
        float(trained_rows[key].passed) - float(baseline_rows[key].passed)
        for key in sorted(baseline_rows)
    ]
    rng = random.Random(seed)
    estimates = [
        sum(rng.choice(deltas) for _ in deltas) / len(deltas)
        for _ in range(bootstrap_rounds)
    ]
    estimates.sort()
    return ComparisonV1(
        baseline_protocol_hash=baseline.protocol.protocol_hash,
        trained_protocol_hash=trained.protocol.protocol_hash,
        compatibility_hash=compatibility_hash,
        compatible=True,
        paired_sample_count=len(deltas),
        baseline_success_rate=baseline.success_rate,
        trained_success_rate=trained.success_rate,
        paired_delta=sum(deltas) / len(deltas),
        paired_ci_low=estimates[int(0.025 * (len(estimates) - 1))],
        paired_ci_high=estimates[int(0.975 * (len(estimates) - 1))],
    )


def _hash(value) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

