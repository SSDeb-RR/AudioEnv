"""Held-out evaluation records and deterministic confidence intervals."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import random
from typing import Iterable

from .adapter import EnvironmentPackage


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


def _hash(value) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

