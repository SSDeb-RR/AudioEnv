"""Minimal Verifiers integration boundary without owning a trainer."""

from __future__ import annotations

from typing import Any

from ..adapter import EnvironmentPackage
from ..runtime import AudioEnvironment


def load_verifiers_taskset(package: EnvironmentPackage) -> list[dict[str, Any]]:
    """Return serializable task rows suitable for a Verifiers DatasetBuilder.

    Hidden facts, expected state, and oracle actions intentionally remain inside
    the runtime and are never included in the task rows.
    """
    return [
        {
            "task_id": task.id,
            "split": task.split,
            "environment_id": package.spec.id,
            "environment_version": package.spec.version,
        }
        for task in package.tasks
    ]


def make_runtime(package: EnvironmentPackage) -> AudioEnvironment:
    return AudioEnvironment(package)

