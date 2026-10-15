"""Pinned τ² retail task-source importer.

This imports provenance, task goals, required actions, and the official split.
Execution remains delegated to an isolated τ² worker; the importer never
pretends generic Python tool code is safe to run in the AudioEnv process.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from typing import Any


TAU2_V1_0_1_COMMIT = "fc0055dc4e0a316c3f83133267fbd6faaa770992"


@dataclass(frozen=True, slots=True)
class Tau2TaskDescriptor:
    id: str
    split: str
    reason_for_call: str
    known_info: str
    required_actions: tuple[dict[str, Any], ...]
    communicate_info: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Tau2ImportManifest:
    source_revision: str
    source_hash: str
    license: str
    domain: str
    tasks: tuple[Tau2TaskDescriptor, ...]


class Tau2RetailSourceAdapter:
    def inspect(self, checkout: Path) -> dict[str, Any]:
        paths = self._paths(checkout)
        missing = [str(path) for path in paths.values() if not path.is_file()]
        revision = _git_revision(checkout)
        return {
            "supported": not missing and revision == TAU2_V1_0_1_COMMIT,
            "domain": "retail",
            "revision": revision,
            "expected_revision": TAU2_V1_0_1_COMMIT,
            "missing": missing,
            "execution_mode": "isolated_tau2_worker_required",
        }

    def import_source(self, checkout: Path) -> Tau2ImportManifest:
        inspection = self.inspect(checkout)
        if not inspection["supported"]:
            raise ValueError(f"unsupported τ² checkout: {inspection}")
        paths = self._paths(checkout)
        tasks = json.loads(paths["tasks"].read_text())
        splits = json.loads(paths["splits"].read_text())
        split_by_id = {str(task_id): split for split in ("train", "test") for task_id in splits[split]}
        descriptors = []
        for task in tasks:
            instructions = task["user_scenario"]["instructions"]
            criteria = task["evaluation_criteria"]
            descriptors.append(
                Tau2TaskDescriptor(
                    id=str(task["id"]),
                    split=split_by_id[str(task["id"])],
                    reason_for_call=str(instructions.get("reason_for_call") or ""),
                    known_info=str(instructions.get("known_info") or ""),
                    required_actions=tuple(criteria.get("actions") or ()),
                    communicate_info=tuple(str(item) for item in (criteria.get("communicate_info") or ())),
                )
            )
        source_hash = sha256(paths["tasks"].read_bytes() + paths["splits"].read_bytes()).hexdigest()
        return Tau2ImportManifest(
            source_revision=inspection["revision"],
            source_hash=source_hash,
            license="MIT",
            domain="retail",
            tasks=tuple(descriptors),
        )

    @staticmethod
    def _paths(checkout: Path) -> dict[str, Path]:
        root = checkout / "data/tau2/domains/retail"
        return {"tasks": root / "tasks.json", "splits": root / "split_tasks.json", "license": checkout / "LICENSE"}


def _git_revision(path: Path) -> str | None:
    process = subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True, check=False)
    return process.stdout.strip() if process.returncode == 0 else None
