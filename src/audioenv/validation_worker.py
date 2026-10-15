"""Constrained JSON-lines worker for declarative environment validation."""

from __future__ import annotations

from dataclasses import asdict
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from .artifacts import LocalArtifactStore
from .authoring import CustomTaskInput, DeterministicEnvironmentAuthor


PROTOCOL_VERSION = "audioenv.validation-worker/v1"
MAX_REQUEST_BYTES = 2 * 1024 * 1024


class ValidationWorkerRequestV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    protocol: Literal["audioenv.validation-worker/v1"] = PROTOCOL_VERSION
    request_id: str
    operation: Literal["validate_custom"]
    custom_task: CustomTaskInput


def handle(request: ValidationWorkerRequestV1) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="audioenv-validation-") as directory:
        authored = DeterministicEnvironmentAuthor(LocalArtifactStore(Path(directory))).author(request.custom_task)
        return {
            "passed": authored.validation.passed,
            "checks": [asdict(check) for check in authored.validation.checks],
            "environment_hash": authored.boundary.content_hash,
            "verifier_hash": authored.boundary.protected_verifier.sha256,
            "task_count": len(authored.package.tasks),
            "worker": "audioenv-declarative-validator-v1",
        }


def main() -> int:
    raw = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
    request_id = None
    try:
        if len(raw) > MAX_REQUEST_BYTES:
            raise ValueError("validation request exceeds size limit")
        request = ValidationWorkerRequestV1.model_validate_json(raw)
        request_id = request.request_id
        response = {
            "protocol": PROTOCOL_VERSION,
            "request_id": request_id,
            "ok": True,
            "result": handle(request),
        }
    except Exception as exc:
        response = {
            "protocol": PROTOCOL_VERSION,
            "request_id": request_id,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    sys.stdout.write(json.dumps(response, sort_keys=True, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
