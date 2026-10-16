"""Import Harbor tasks without exposing or executing their private verifier."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import tomllib
from typing import Any

from ..artifacts import LocalArtifactStore
from ..contracts import (
    AuthoringStatus,
    EnvironmentPackageV1,
    EnvironmentType,
    ProtectedVerifierRef,
    PublicTaskV1,
    SourceProvenance,
)


@dataclass(frozen=True, slots=True)
class HarborTaskBundle:
    task_id: str
    root: Path
    instruction: str
    dockerfile: Path
    verifier: Path
    solution: Path | None
    metadata: dict[str, Any]
    content_hash: str


class HarborEnvironmentAdapter:
    """Convert a Harbor directory into AudioEnv's public/protected boundary."""

    def __init__(self, artifacts: LocalArtifactStore):
        self.artifacts = artifacts

    def inspect(self, source: Path) -> dict[str, Any]:
        try:
            tasks = self._task_roots(source)
        except ValueError as exc:
            return {"supported": False, "reason": str(exc), "task_count": 0}
        return {
            "supported": bool(tasks),
            "task_count": len(tasks),
            "task_ids": [path.name for path in tasks],
            "private_verifiers": all((path / "tests" / "test.sh").is_file() for path in tasks),
        }

    def convert(
        self,
        source: Path,
        *,
        environment_id: str,
        version: str = "1.0.0",
        source_uri: str | None = None,
        revision: str | None = None,
    ) -> tuple[EnvironmentPackageV1, tuple[HarborTaskBundle, ...]]:
        bundles = tuple(self._load_task(path) for path in self._task_roots(source))
        if not bundles:
            raise ValueError("Harbor source contains no tasks")
        verifier_index = self.artifacts.put_json({
            "schema_version": "audioenv.harbor-verifiers/v1",
            "tasks": [{
                "id": item.task_id,
                "verifier_path": str(item.verifier.relative_to(item.root)),
                "verifier_sha256": sha256(item.verifier.read_bytes()).hexdigest(),
                "content_hash": item.content_hash,
            } for item in bundles],
        })
        public_tasks = tuple(PublicTaskV1(
            id=item.task_id,
            split=_split(item.metadata),
            goal=item.instruction,
            tool_names=("terminal", "filesystem"),
        ) for item in bundles)
        payload: dict[str, Any] = {
            "schema_version": "audioenv.environment/v1",
            "id": environment_id,
            "version": version,
            "description": f"Harbor environment imported from {source_uri or source}",
            "environment_type": EnvironmentType.CODING,
            "source": SourceProvenance(
                kind="repository", uri=source_uri or str(source), revision=revision,
                license=str(bundles[0].metadata.get("license") or "source-dependent"),
            ).model_dump(mode="json"),
            "required_capabilities": ("terminal", "filesystem", "oci-sandbox"),
            "tasks": [item.model_dump(mode="json") for item in public_tasks],
            "protected_verifier": ProtectedVerifierRef(
                artifact_id=verifier_index.id, sha256=verifier_index.sha256,
            ).model_dump(mode="json"),
            "authoring_status": AuthoringStatus.GENERATED_NEEDS_REVIEW,
            "parent_hash": None,
        }
        payload["content_hash"] = EnvironmentPackageV1.hash_payload(payload)
        return EnvironmentPackageV1.model_validate(payload), bundles

    @staticmethod
    def _task_roots(source: Path) -> tuple[Path, ...]:
        source = source.resolve()
        if not source.is_dir():
            raise ValueError("Harbor source must be a directory")
        if (source / "task.toml").is_file():
            candidates = (source,)
        else:
            candidates = tuple(sorted(path.parent for path in source.glob("*/task.toml")))
        for path in candidates:
            for required in ("instruction.md", "environment/Dockerfile", "tests/test.sh"):
                candidate = (path / required).resolve()
                if path not in candidate.parents or not candidate.is_file() or candidate.is_symlink():
                    raise ValueError(f"invalid Harbor task {path.name}: missing safe {required}")
        return candidates

    @staticmethod
    def _load_task(root: Path) -> HarborTaskBundle:
        metadata = tomllib.loads((root / "task.toml").read_text(encoding="utf-8"))
        instruction = (root / "instruction.md").read_text(encoding="utf-8")
        dockerfile = root / "environment" / "Dockerfile"
        verifier = root / "tests" / "test.sh"
        solutions = tuple(
            path for path in sorted((root / "solution").glob("*")) if path.is_file()
        ) if (root / "solution").is_dir() else ()
        content: list[bytes] = []
        total_size = 0
        task_entries = tuple(sorted(root.rglob("*")))
        if any(path.is_symlink() for path in task_entries):
            raise ValueError("Harbor task trees cannot contain symbolic links")
        for path in (item for item in task_entries if item.is_file()):
            if root not in path.resolve().parents:
                raise ValueError(f"unsafe Harbor task file: {path.relative_to(root)}")
            payload = path.read_bytes()
            if len(payload) > 16 * 1024 * 1024:
                raise ValueError(f"Harbor task file exceeds 16 MiB: {path.relative_to(root)}")
            total_size += len(payload)
            if total_size > 256 * 1024 * 1024:
                raise ValueError("Harbor task exceeds 256 MiB")
            content.extend((path.relative_to(root).as_posix().encode(), payload))
        return HarborTaskBundle(
            task_id=str(metadata.get("id") or root.name), root=root,
            instruction=instruction, dockerfile=dockerfile, verifier=verifier,
            solution=solutions[0] if solutions else None, metadata=metadata,
            content_hash=sha256(b"\0".join(content)).hexdigest(),
        )


def _split(metadata: dict[str, Any]) -> str:
    value = str(metadata.get("split") or metadata.get("task", {}).get("split") or "train")
    return value if value in {"train", "validation", "test"} else "train"
