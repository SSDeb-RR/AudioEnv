"""Pinned Repo2RLEnv generator executed through a sandbox provider."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import subprocess
from typing import Protocol

from ..contracts import SandboxPolicyV1


REPO2RLENV_COMMIT = "e5e6a93a6d7947a012a361b68457a982cc85fc9e"
REPO2RLENV_VERSION = "0.9.3"


@dataclass(frozen=True, slots=True)
class SandboxResult:
    returncode: int
    stdout: str
    stderr: str


class SandboxExecutor(Protocol):
    def run(
        self, image: str, command: tuple[str, ...], *, source: Path,
        output: Path, policy: SandboxPolicyV1, bootstrap_network: bool,
    ) -> SandboxResult: ...


class DockerSandboxExecutor:
    """Small, auditable Docker boundary; never mounts the Docker socket."""

    def probe(self) -> tuple[str, ...]:
        reasons = []
        if shutil.which("docker") is None:
            reasons.append("docker is unavailable")
        return tuple(reasons)

    def run(
        self, image: str, command: tuple[str, ...], *, source: Path,
        output: Path, policy: SandboxPolicyV1, bootstrap_network: bool,
    ) -> SandboxResult:
        reasons = self.probe()
        if reasons:
            raise RuntimeError("; ".join(reasons))
        output.mkdir(parents=True, exist_ok=True)
        args = [
            "docker", "run", "--rm", "--init", "--read-only",
            "--user", "65532:65532", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--pids-limit", str(policy.pid_limit), "--cpus", str(policy.cpu_limit),
            "--memory", f"{policy.memory_mb}m", "--network", "bridge" if bootstrap_network else "none",
            "--mount", f"type=bind,src={source.resolve()},dst=/source,readonly",
            "--mount", f"type=bind,src={output.resolve()},dst=/output",
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=1g", image, *command,
        ]
        process = subprocess.run(
            args, capture_output=True, text=True, timeout=policy.timeout_seconds, check=False,
            env={"PATH": shutil.which("docker") or "docker"},
        )
        return SandboxResult(process.returncode, process.stdout, process.stderr)


class Repo2RLEnvGenerator:
    # The official package is invoked in an isolated Python 3.12 uv image.
    # Runtime pipelines need Docker; AudioEnv will not mount the host socket.
    image = "ghcr.io/astral-sh/uv:python3.12-bookworm-slim"

    def __init__(self, executor: SandboxExecutor):
        self.executor = executor

    def generate(
        self, repository: Path, output: Path, *, pipeline: str = "pr_diff",
        limit: int = 5, llm: str | None = None, policy: SandboxPolicyV1 | None = None,
    ) -> dict[str, object]:
        if pipeline not in {"pr_runtime", "commit_runtime", "pr_diff"}:
            raise ValueError("unsupported Repo2RLEnv pipeline")
        if pipeline != "pr_diff":
            raise RuntimeError(
                f"{pipeline} requires a nested/sibling Docker executor; AudioEnv refuses to mount "
                "the host Docker socket. Use pr_diff or configure a future sandbox provider."
            )
        if limit < 1 or limit > 100:
            raise ValueError("limit must be between 1 and 100")
        policy = policy or SandboxPolicyV1(network="bootstrap-only")
        output.mkdir(parents=True, exist_ok=True)
        command = [
            "uvx", "--from", f"repo2rlenv=={REPO2RLENV_VERSION}",
            "repo2rlenv", "--no-ui", "generate", "--repo", "/source", "--pipeline", pipeline,
            "--pipeline-opt", f"limit={limit}", "--out", "/output",
        ]
        if llm:
            command.extend(("--llm", llm))
        result = self.executor.run(
            self.image, tuple(command), source=repository, output=output,
            policy=policy, bootstrap_network=True,
        )
        report = {
            "schema_version": "audioenv.repo2rlenv-run/v1", "version": REPO2RLENV_VERSION,
            "commit": REPO2RLENV_COMMIT, "pipeline": pipeline, "returncode": result.returncode,
            "stdout_tail": result.stdout[-4_000:], "stderr_tail": result.stderr[-4_000:],
        }
        (output / "audioenv-repo2rlenv-run.json").write_text(
            json.dumps(report, sort_keys=True, indent=2), encoding="utf-8",
        )
        if result.returncode:
            raise RuntimeError(f"Repo2RLEnv generation failed: {result.stderr[-1000:]}")
        return report
