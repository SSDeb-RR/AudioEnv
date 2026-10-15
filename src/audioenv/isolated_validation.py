"""Control-plane client for the declarative validation subprocess."""

from __future__ import annotations

import json
import os
import resource
import subprocess
import sys
from typing import Any
from uuid import uuid4

from .authoring import CustomTaskInput
from .validation_worker import PROTOCOL_VERSION


class IsolatedValidationError(RuntimeError):
    pass


class IsolatedValidationRunner:
    def __init__(self, *, timeout_seconds: float = 45.0, max_memory_mb: int = 1024):
        if timeout_seconds <= 0 or max_memory_mb < 256:
            raise ValueError("invalid validation worker resource limits")
        self.timeout_seconds = timeout_seconds
        self.max_memory_mb = max_memory_mb

    def validate_custom(self, custom_task: CustomTaskInput) -> dict[str, Any]:
        request_id = str(uuid4())
        request = {
            "protocol": PROTOCOL_VERSION,
            "request_id": request_id,
            "operation": "validate_custom",
            "custom_task": custom_task.model_dump(mode="json"),
        }
        environment = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONNOUSERSITE": "1",
            "PYTHONHASHSEED": "0",
            "LC_ALL": "C",
        }
        try:
            process = subprocess.run(
                (sys.executable, "-m", "audioenv.validation_worker"),
                input=json.dumps(request, sort_keys=True, separators=(",", ":")),
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=self.timeout_seconds,
                env=environment,
                start_new_session=True,
                preexec_fn=self._resource_limits if os.name == "posix" else None,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise IsolatedValidationError("validation worker timed out") from exc
        if process.returncode:
            raise IsolatedValidationError(f"validation worker exited with status {process.returncode}")
        try:
            response = json.loads(process.stdout)
        except json.JSONDecodeError as exc:
            raise IsolatedValidationError("validation worker emitted invalid JSON") from exc
        if response.get("protocol") != PROTOCOL_VERSION or response.get("request_id") != request_id:
            raise IsolatedValidationError("validation worker response correlation failed")
        if response.get("ok") is not True:
            raise IsolatedValidationError(str(response.get("error") or "validation worker failed"))
        result = response.get("result")
        if not isinstance(result, dict):
            raise IsolatedValidationError("validation worker result must be an object")
        return result

    def _resource_limits(self) -> None:
        memory = self.max_memory_mb * 1024 * 1024
        limits = (
            (resource.RLIMIT_AS, memory),
            (resource.RLIMIT_CPU, max(1, int(self.timeout_seconds))),
            (resource.RLIMIT_NOFILE, 64),
        )
        for kind, requested in limits:
            try:
                _, hard = resource.getrlimit(kind)
                effective = requested if hard == resource.RLIM_INFINITY else min(requested, hard)
                resource.setrlimit(kind, (effective, effective))
            except (OSError, ValueError):
                # The worker remains isolated by process, deadline, sanitized
                # environment, and closed descriptors when a host does not
                # implement a particular POSIX resource limit.
                continue
