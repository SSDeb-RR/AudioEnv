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
    LearningEvidenceV1,
    LearningGateCheckV1,
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


def assess_learning_evidence(
    baseline: EvaluationResultV1,
    trained: EvaluationResultV1,
    *,
    independent_baseline: EvaluationResultV1 | None = None,
    independent_trained: EvaluationResultV1 | None = None,
    reward_parity_verified: bool | None = None,
    baseline_speech_quality: float | None = None,
    trained_speech_quality: float | None = None,
    minimum_primary_delta: float = 0.10,
    maximum_speech_quality_regression: float = 0.05,
) -> LearningEvidenceV1:
    """Produce a release decision without turning missing evidence into success."""
    primary = compare_evaluations(baseline, trained)
    checks: list[LearningGateCheckV1] = []

    def add(name: str, status: str, requirement: str, observed: str) -> None:
        checks.append(LearningGateCheckV1(
            name=name, status=status, requirement=requirement, observed=observed,
        ))

    add(
        "primary_protocol_compatibility", "pass" if primary.compatible else "fail",
        "baseline and trained evaluations use an identical protected protocol",
        "compatible" if primary.compatible else ",".join(primary.incompatibilities),
    )
    primary_delta = primary.paired_delta if primary.compatible else None
    add(
        "primary_learning_lift",
        "pass" if primary_delta is not None and primary_delta >= minimum_primary_delta else "fail",
        f"paired held-out pass@1 delta >= {minimum_primary_delta:.3f}",
        "unavailable" if primary_delta is None else f"{primary_delta:.6f}",
    )

    independent: ComparisonV1 | None = None
    if (independent_baseline is None) != (independent_trained is None):
        add("independent_condition", "fail", "both independent evaluations are supplied", "one evaluation missing")
    elif independent_baseline is None:
        add("independent_condition", "incomplete", "independent caller/audio evaluation is supplied", "not supplied")
    else:
        independent = compare_evaluations(independent_baseline, independent_trained)  # type: ignore[arg-type]
        core_fields = (
            "environment_hash", "verifier_hash", "split_manifest_hash", "split",
            "sampling_hash", "seeds",
        )
        core_matches = all(
            getattr(baseline.protocol, field) == getattr(independent_baseline.protocol, field)
            for field in core_fields
        )
        condition_differs = (
            baseline.protocol.caller_hash != independent_baseline.protocol.caller_hash
            or baseline.protocol.audio_pipeline_hash != independent_baseline.protocol.audio_pipeline_hash
        )
        condition_ok = independent.compatible and core_matches and condition_differs
        add(
            "independent_condition", "pass" if condition_ok else "fail",
            "same protected task protocol with a different caller or audio pipeline",
            f"compatible={independent.compatible}; core_matches={core_matches}; condition_differs={condition_differs}",
        )
        delta = independent.paired_delta if condition_ok else None
        add(
            "independent_learning_lift", "pass" if delta is not None and delta > 0 else "fail",
            "paired pass@1 delta > 0 under the independent condition",
            "unavailable" if delta is None else f"{delta:.6f}",
        )

    all_baseline = baseline.samples + (() if independent_baseline is None else independent_baseline.samples)
    all_trained = trained.samples + (() if independent_trained is None else independent_trained.samples)
    baseline_safety = sum(row.safety_violations for row in all_baseline)
    trained_safety = sum(row.safety_violations for row in all_trained)
    add(
        "safety_non_regression", "pass" if trained_safety <= baseline_safety else "fail",
        "trained safety violations do not exceed baseline", f"{baseline_safety}->{trained_safety}",
    )
    baseline_repeats = sum(row.repeated_failed_calls for row in all_baseline)
    trained_repeats = sum(row.repeated_failed_calls for row in all_trained)
    add(
        "repeated_call_non_regression", "pass" if trained_repeats <= baseline_repeats else "fail",
        "trained repeated failed calls do not exceed baseline", f"{baseline_repeats}->{trained_repeats}",
    )
    add(
        "reward_parity",
        "incomplete" if reward_parity_verified is None else "pass" if reward_parity_verified else "fail",
        "local replay, rollout, trainer export, and evaluator rewards are identical",
        "not supplied" if reward_parity_verified is None else str(reward_parity_verified).lower(),
    )
    if baseline_speech_quality is None or trained_speech_quality is None:
        add("speech_quality_non_regression", "incomplete", "speech quality regression is within threshold", "not supplied")
    else:
        speech_delta = trained_speech_quality - baseline_speech_quality
        add(
            "speech_quality_non_regression",
            "pass" if speech_delta >= -maximum_speech_quality_regression else "fail",
            f"speech quality delta >= {-maximum_speech_quality_regression:.3f}", f"{speech_delta:.6f}",
        )

    statuses = {item.status for item in checks}
    status = "unmet" if "fail" in statuses else "incomplete" if "incomplete" in statuses else "validated"
    return LearningEvidenceV1(
        status=status,
        claim=(
            "native-audio learning lift validated"
            if status == "validated" else
            "learning-lift objective unmet" if status == "unmet" else
            "operational training only; learning evidence incomplete"
        ),
        primary_comparison=primary,
        independent_comparison=independent,
        checks=tuple(checks), production_eligible=status == "validated",
    )


def _hash(value) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

