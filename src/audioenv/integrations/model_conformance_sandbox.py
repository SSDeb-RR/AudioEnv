"""Rootless OCI boundary for executable user-supplied model drivers."""

from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import subprocess
from typing import Protocol

from ..contracts import ModelConformanceReportV1
from ..models.omni_driver import ModelConformanceRunner


_IMMUTABLE_IMAGE = re.compile(r"^[^\s@]+@sha256:[0-9a-f]{64}$")


class ModelConformanceTransport(Protocol):
    def run(
        self, *, package_path: Path, model_id: str, image: str,
        checkpoint_path: Path | None = None,
    ) -> ModelConformanceReportV1: ...


class DockerModelConformanceTransport:
    """Execute the conformance entrypoint with no network, secrets, or host socket."""

    def probe(self) -> tuple[str, ...]:
        docker = shutil.which("docker")
        if docker is None:
            return ("docker is unavailable",)
        process = subprocess.run(
            [docker, "info", "--format", "{{json .SecurityOptions}}"],
            capture_output=True, text=True, timeout=15, check=False,
        )
        if process.returncode:
            return ("docker daemon is unavailable",)
        if "rootless" not in process.stdout.lower():
            return ("rootless Docker is required for uploaded driver execution",)
        return ()

    def run(
        self, *, package_path: Path, model_id: str, image: str,
        checkpoint_path: Path | None = None,
    ) -> ModelConformanceReportV1:
        if _IMMUTABLE_IMAGE.fullmatch(image) is None:
            raise ValueError("conformance image must use an immutable sha256 digest")
        reasons = self.probe()
        if reasons:
            raise RuntimeError("; ".join(reasons))
        package_path = package_path.resolve(strict=True)
        command = [
            shutil.which("docker") or "docker", "run", "--rm", "--init",
            "--network", "none", "--read-only", "--user", "65532:65532",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--pids-limit", "256", "--cpus", "4", "--memory", "32g",
            "--gpus", "all",
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=2g",
            "--mount", f"type=bind,src={package_path},dst=/input/driver.zip,readonly",
        ]
        if checkpoint_path is not None:
            checkpoint = checkpoint_path.resolve(strict=True)
            command.extend((
                "--mount", f"type=bind,src={checkpoint},dst=/model,readonly",
                "--env", "AUDIOENV_MODEL_PATH=/model",
            ))
        command.extend((
            image, "audioenv-model-conformance-worker", "--package", "/input/driver.zip",
            "--model-id", model_id,
        ))
        try:
            process = subprocess.run(
                command, capture_output=True, text=True, timeout=1800, check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("model conformance exceeded 30 minutes") from exc
        line = next((row for row in reversed(process.stdout.splitlines()) if row.strip()), "")
        try:
            response = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError("model conformance worker returned invalid JSON") from exc
        if process.returncode or not response.get("ok"):
            raise RuntimeError(str(response.get("error") or "model conformance failed")[:1000])
        return ModelConformanceReportV1.model_validate(response.get("report"))


class ModelConformanceSandbox:
    def __init__(self, transport: ModelConformanceTransport):
        self.transport = transport

    def validate(
        self, *, package_path: Path, model_id: str, image: str,
        checkpoint_path: Path | None = None,
    ) -> ModelConformanceReportV1:
        report = self.transport.run(
            package_path=package_path, model_id=model_id, image=image,
            checkpoint_path=checkpoint_path,
        )
        if report.model_id != model_id:
            raise RuntimeError("conformance report model_id does not match the registry model")
        expected = set(ModelConformanceRunner.check_names)
        actual = {item.name for item in report.checks}
        if actual != expected:
            raise RuntimeError("conformance report does not contain the exact required gates")
        return report
