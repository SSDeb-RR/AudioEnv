"""Portable, leakage-safe package and trace exports."""

from __future__ import annotations

import json
from pathlib import Path

from .adapter import EnvironmentPackage
from .spec import public_dataclass
from .trace import EpisodeTrace


def export_package(package: EnvironmentPackage, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    public_payload = {
        "format": "audioenv/v1",
        "environment": public_dataclass(package.spec),
        "tasks": [
            {"id": task.id, "split": task.split, "user_goal": task.user_goal}
            for task in package.tasks
        ],
    }
    output = destination / "environment.json"
    output.write_text(json.dumps(public_payload, sort_keys=True, indent=2), encoding="utf-8")
    private = destination / ".audioenv-private"
    private.mkdir(exist_ok=True)
    verifier_payload = {
        "format": "audioenv/verifier-v1",
        "tasks": [public_dataclass(task) for task in package.tasks],
    }
    (private / "verifier.json").write_text(
        json.dumps(verifier_payload, sort_keys=True, indent=2), encoding="utf-8"
    )
    return output


def export_trace(trace: EpisodeTrace, destination: Path, *, include_observability: bool = False) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(trace.to_json(include_observability=include_observability), encoding="utf-8")
    return destination

