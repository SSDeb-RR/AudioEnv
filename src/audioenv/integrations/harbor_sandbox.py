"""Fail-closed OCI validation for imported Harbor tasks.

The validator deliberately exposes only pass/fail evidence and hashes. Private
verifier output and reference solutions never cross this boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from pathlib import PurePosixPath
import shlex
import shutil
import subprocess
import re
from typing import Protocol

from ..adapters.harbor import HarborTaskBundle
from ..contracts import EnvironmentPackageV1, SandboxPolicyV1


@dataclass(frozen=True, slots=True)
class HarborTrialResult:
    exit_code: int
    output_hash: str


class HarborContainerTransport(Protocol):
    def probe(self) -> tuple[str, ...]: ...

    def build(self, *, task: HarborTaskBundle, image_tag: str, policy: SandboxPolicyV1) -> None: ...

    def trial(
        self, *, task: HarborTaskBundle, image_tag: str, mode: str,
        policy: SandboxPolicyV1,
    ) -> HarborTrialResult: ...


class DockerHarborTransport:
    """Rootless-Docker transport with no runtime network or host socket."""

    def probe(self) -> tuple[str, ...]:
        docker = shutil.which("docker")
        if docker is None:
            return ("docker is unavailable",)
        try:
            process = subprocess.run(
                [docker, "info", "--format", "{{json .SecurityOptions}}"],
                capture_output=True, text=True, timeout=15, check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return (f"docker probe failed: {exc}",)
        if process.returncode:
            return ("docker daemon is unavailable",)
        if "rootless" not in process.stdout.lower():
            return ("rootless Docker is required for untrusted Harbor validation",)
        return ()

    def build(self, *, task: HarborTaskBundle, image_tag: str, policy: SandboxPolicyV1) -> None:
        reasons = self.probe()
        if reasons:
            raise RuntimeError("; ".join(reasons))
        docker = shutil.which("docker") or "docker"
        process = subprocess.run(
            [docker, "build", "--network", "bridge" if policy.network == "bootstrap-only" else "none", "--tag", image_tag,
             "--file", str(task.dockerfile), str(task.dockerfile.parent)],
            capture_output=True, text=True, timeout=policy.timeout_seconds, check=False,
        )
        if process.returncode:
            raise RuntimeError("Harbor image build failed")

    def trial(
        self, *, task: HarborTaskBundle, image_tag: str, mode: str,
        policy: SandboxPolicyV1,
    ) -> HarborTrialResult:
        if mode not in {"oracle", "noop", "adversarial"}:
            raise ValueError("unsupported Harbor validation trial")
        reasons = self.probe()
        if reasons:
            raise RuntimeError("; ".join(reasons))
        fixture = task.solution if mode == "oracle" else _adversarial_fixture(task) if mode == "adversarial" else None
        if mode != "noop" and fixture is None:
            raise ValueError(f"{mode} fixture is unavailable")
        audioenv = task.metadata.get("audioenv", {})
        if not isinstance(audioenv, dict):
            raise ValueError("task.toml [audioenv] must be a table")
        workspace = str(audioenv.get("workspace") or "/workspace")
        source_dir = str(audioenv.get("source_dir") or "/app")
        if not PurePosixPath(workspace).is_absolute() or not PurePosixPath(source_dir).is_absolute():
            raise ValueError("Harbor workspace and source_dir must be absolute container paths")
        verifier_command = str(audioenv.get("verifier_command") or "sh /tests/test.sh")
        fixture_command = str(
            audioenv.get(f"{mode}_command")
            or (f"sh /fixture/{fixture.name}" if fixture is not None else "true")
        )
        command = (
            f"set -eu; cp -a {shlex.quote(source_dir)}/. {shlex.quote(workspace)}/; "
            f"{fixture_command}; {verifier_command}"
        )
        docker = shutil.which("docker") or "docker"
        args = [
            docker, "run", "--rm", "--init", "--network", "none", "--read-only",
            "--user", "65532:65532", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--pids-limit", str(policy.pid_limit), "--cpus", str(policy.cpu_limit),
            "--memory", f"{policy.memory_mb}m", "--workdir", workspace,
            "--mount", f"type=bind,src={task.verifier.parent.resolve()},dst=/tests,readonly",
            "--tmpfs", f"{workspace}:rw,nosuid,size={max(64, policy.memory_mb // 2)}m",
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=256m",
        ]
        if fixture is not None:
            args.extend(("--mount", f"type=bind,src={fixture.parent.resolve()},dst=/fixture,readonly"))
        args.extend((image_tag, "sh", "-lc", command))
        try:
            process = subprocess.run(
                args, capture_output=True, timeout=policy.timeout_seconds, check=False,
            )
        except subprocess.TimeoutExpired as exc:
            payload = (exc.stdout or b"") + b"\0" + (exc.stderr or b"") + b"\0timeout"
            return HarborTrialResult(124, sha256(payload).hexdigest())
        payload = process.stdout + b"\0" + process.stderr
        return HarborTrialResult(process.returncode, sha256(payload).hexdigest())


class HarborSandboxValidator:
    def __init__(self, transport: HarborContainerTransport):
        self.transport = transport

    def validate(
        self, package: EnvironmentPackageV1, bundles: tuple[HarborTaskBundle, ...],
        *, policy: SandboxPolicyV1 | None = None,
    ) -> dict[str, object]:
        policy = policy or SandboxPolicyV1()
        reasons = self.transport.probe()
        if reasons:
            return _report(package, (), passed=False, diagnostics=reasons)
        checks: list[dict[str, object]] = []
        verifier_before = _verifier_hash(bundles)
        for task in bundles:
            image_tag = f"audioenv-harbor:{task.content_hash[:24]}"
            try:
                source_before = _task_tree_hash(task)
            except (OSError, ValueError) as exc:
                checks.extend((
                    _check(f"{task.task_id}:source_immutability", False, str(exc)),
                    _check(f"{task.task_id}:sandbox_build", False, "task source is unsafe"),
                ))
                continue
            source_bound = source_before == task.content_hash
            checks.append(_check(f"{task.task_id}:source_immutability", source_bound))
            if not source_bound:
                checks.append(_check(
                    f"{task.task_id}:sandbox_build", False,
                    "task source changed after environment import",
                ))
                continue
            try:
                _validate_build_definition(task)
                self.transport.build(task=task, image_tag=image_tag, policy=policy)
                checks.append(_check(f"{task.task_id}:sandbox_build", True))
                oracle_a = self.transport.trial(task=task, image_tag=image_tag, mode="oracle", policy=policy)
                oracle_b = self.transport.trial(task=task, image_tag=image_tag, mode="oracle", policy=policy)
                noop_a = self.transport.trial(task=task, image_tag=image_tag, mode="noop", policy=policy)
                noop_b = self.transport.trial(task=task, image_tag=image_tag, mode="noop", policy=policy)
                if _adversarial_fixture(task) is None:
                    raise ValueError("adversarial fixture is unavailable")
                adversarial = self.transport.trial(task=task, image_tag=image_tag, mode="adversarial", policy=policy)
                checks.extend((
                    _check(f"{task.task_id}:oracle_pass", oracle_a.exit_code == 0),
                    _check(f"{task.task_id}:noop_fail", noop_a.exit_code != 0),
                    _check(f"{task.task_id}:reward_mutation", adversarial.exit_code != 0),
                    _check(f"{task.task_id}:deterministic_replay", oracle_a == oracle_b and noop_a == noop_b),
                    _check(f"{task.task_id}:state_isolation", noop_a == noop_b),
                    _check(f"{task.task_id}:no_text_leak", _public_boundary_is_clean(package, task)),
                    _check(
                        f"{task.task_id}:source_immutability_after",
                        _task_tree_hash(task) == task.content_hash,
                    ),
                ))
            except (OSError, RuntimeError, ValueError) as exc:
                checks.append(_check(f"{task.task_id}:sandbox_build", False, str(exc)))
        verifier_after = _verifier_hash(bundles)
        checks.extend((
            _check("verifier_immutability", verifier_before == verifier_after),
            _check("reward_integrity", _reward_integrity(checks)),
        ))
        passed = all(bool(row["passed"]) for row in checks)
        return _report(
            package, tuple(checks), passed=passed,
            diagnostics=() if passed else ("one or more protected sandbox gates failed",),
            verifier_before=verifier_before, verifier_after=verifier_after,
        )


def _adversarial_fixture(task: HarborTaskBundle) -> Path | None:
    folder = task.root / "adversarial"
    if not folder.is_dir():
        return None
    return next((path for path in sorted(folder.iterdir()) if path.is_file() and not path.is_symlink()), None)


def _validate_build_definition(task: HarborTaskBundle) -> None:
    """Reject non-reproducible or secret-bearing image build primitives."""
    dockerfile = task.dockerfile.read_text(encoding="utf-8", errors="strict")
    logical_lines = re.sub(r"\\\r?\n", " ", dockerfile).splitlines()
    from_count = 0
    for number, raw in enumerate(logical_lines, 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        instruction, _, value = line.partition(" ")
        instruction = instruction.upper()
        value = value.strip()
        if instruction == "FROM":
            from_count += 1
            parts = [part for part in value.split() if not part.startswith("--platform=")]
            image = parts[0] if parts else ""
            if image.casefold() != "scratch" and not re.fullmatch(
                r"[^\s@]+@sha256:[0-9a-f]{64}", image,
            ):
                raise ValueError(
                    f"Dockerfile line {number} uses a mutable base image; pin FROM by sha256"
                )
        if instruction == "ADD" and re.search(
            r"(?:https?|git|ssh)://|git@", value, flags=re.IGNORECASE,
        ):
            raise ValueError(f"Dockerfile line {number} uses a remote ADD source")
        if instruction == "RUN" and re.search(
            r"--mount\s*=\s*type\s*=\s*(?:secret|ssh|bind)\b",
            value, flags=re.IGNORECASE,
        ):
            raise ValueError(f"Dockerfile line {number} requests a forbidden build mount")
    if from_count == 0:
        raise ValueError("Dockerfile requires at least one FROM instruction")


def _verifier_hash(bundles: tuple[HarborTaskBundle, ...]) -> str:
    return sha256(b"\0".join(task.verifier.read_bytes() for task in bundles)).hexdigest()


def _task_tree_hash(task: HarborTaskBundle) -> str:
    content: list[bytes] = []
    for path in sorted(task.root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Harbor task source gained a symbolic link after import")
        if path.is_file():
            content.extend((path.relative_to(task.root).as_posix().encode(), path.read_bytes()))
    return sha256(b"\0".join(content)).hexdigest()


def _public_boundary_is_clean(package: EnvironmentPackageV1, task: HarborTaskBundle) -> bool:
    public = package.model_dump_json()
    secrets = [task.verifier.read_text(encoding="utf-8", errors="replace")]
    if task.solution is not None:
        secrets.append(task.solution.read_text(encoding="utf-8", errors="replace"))
    fragments = {
        line.strip() for value in secrets for line in value.splitlines()
        if len(line.strip()) >= 12 and not line.lstrip().startswith("#!")
    }
    return all(fragment not in public for fragment in fragments)


def _reward_integrity(checks: list[dict[str, object]]) -> bool:
    relevant = [row for row in checks if str(row["name"]).endswith((":oracle_pass", ":noop_fail", ":reward_mutation", ":deterministic_replay"))]
    return bool(relevant) and all(bool(row["passed"]) for row in relevant)


def _check(name: str, passed: bool, reason: str | None = None) -> dict[str, object]:
    result: dict[str, object] = {"name": name, "passed": passed}
    if reason:
        result["reason"] = reason[:500]
    return result


def _report(
    package: EnvironmentPackageV1, checks: tuple[dict[str, object], ...], *,
    passed: bool, diagnostics: tuple[str, ...], verifier_before: str | None = None,
    verifier_after: str | None = None,
) -> dict[str, object]:
    return {
        "schema_version": "audioenv.harbor-validation/v1",
        "environment_hash": package.content_hash,
        "verifier_hash_before": verifier_before or package.protected_verifier.sha256,
        "verifier_hash_after": verifier_after or package.protected_verifier.sha256,
        "passed": passed,
        "checks": list(checks),
        "diagnostics": list(diagnostics),
        "private_output_redacted": True,
    }
