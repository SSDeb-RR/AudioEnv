"""Local subprocess execution provider."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys

from ..contracts import ResourceEstimate


@dataclass(frozen=True, slots=True)
class SubmittedProcess:
    job_id: str
    pid: int
    process: subprocess.Popen


class LocalExecutionProvider:
    provider_id = "local-subprocess"

    def preflight(self) -> ResourceEstimate:
        return ResourceEstimate(
            provider=self.provider_id,
            device_type="cpu",
            device_count=1,
            available=True,
            reasons=(),
        )

    def submit(self, *, database: Path, job_id: str) -> SubmittedProcess:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "audioenv.platform.worker",
                "--database",
                str(database.resolve()),
                "--job-id",
                job_id,
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return SubmittedProcess(job_id, process.pid, process)

