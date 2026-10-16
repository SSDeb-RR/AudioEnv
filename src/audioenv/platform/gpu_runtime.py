"""Fail-closed attestation for the release GPU worker runtime."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
from typing import Any
from importlib import metadata

from ..contracts import GPUWorkerExperimentV1
from ..training.verl_omni import PINNED_VERL_OMNI_COMMIT
from ..training.vllm_audio_bridge import PINNED_VLLM_OMNI_COMMIT


LOCK_SCHEMA = "audioenv.gpu-worker-dependencies/v1"
DEFAULT_LOCK_PATH = Path("/opt/audioenv/dependencies.lock.json")
DEFAULT_PYTHON_ENVIRONMENT_PATH = Path("/opt/audioenv/python-environment.lock")
REQUIRED_LOCK_VALUES = {
    "verl_omni_commit": PINNED_VERL_OMNI_COMMIT,
    "vllm_omni_commit": PINNED_VLLM_OMNI_COMMIT,
    "tau2_commit": "fc0055dc4e0a316c3f83133267fbd6faaa770992",
    "qwen_model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
}


@dataclass(frozen=True, slots=True)
class RuntimeAttestation:
    available: bool
    reasons: tuple[str, ...]
    evidence: dict[str, Any]


def attest_qwen_runtime(
    experiment: GPUWorkerExperimentV1,
    *,
    lock_path: Path | None = None,
    python_environment_path: Path | None = None,
    probe_cuda: bool = True,
    probe_packages: bool = True,
) -> RuntimeAttestation:
    """Validate the immutable image, experiment inputs, and CUDA runtime.

    This runs inside the worker before either the caller service or VeRL is
    started. It deliberately accumulates diagnostics so a failed remote job is
    actionable while still refusing to train on an unattested input.
    """

    reasons: list[str] = []
    evidence: dict[str, Any] = {}
    selected_lock = lock_path or Path(os.environ.get("AUDIOENV_DEPENDENCY_LOCK", DEFAULT_LOCK_PATH))
    lock = _load_lock(selected_lock, reasons)
    selected_python_environment = python_environment_path or Path(
        os.environ.get("AUDIOENV_PYTHON_ENVIRONMENT_LOCK", DEFAULT_PYTHON_ENVIRONMENT_PATH)
    )
    python_environment = _attest_python_environment(selected_python_environment, reasons)
    if python_environment:
        evidence["python_environment_lock"] = python_environment
    if lock:
        evidence["dependency_lock_sha256"] = sha256(selected_lock.read_bytes()).hexdigest()
        evidence["dependency_lock"] = lock
        if lock.get("schema_version") != LOCK_SCHEMA:
            reasons.append(f"dependency lock schema must be {LOCK_SCHEMA}")
        for key, expected in REQUIRED_LOCK_VALUES.items():
            if lock.get(key) != expected:
                reasons.append(f"dependency lock {key} must equal {expected}")
        for key, expected in lock.items():
            if key in {"schema_version", "base_image_policy"}:
                continue
            actual = experiment.dependency_pins.get(key)
            if actual != str(expected):
                reasons.append(f"experiment dependency pin {key} must equal image lock value {expected}")

    model_revision = _model_revision(Path(experiment.model_path or ""))
    evidence["model_revision"] = model_revision
    caller_revision = _model_revision(Path(experiment.caller_model_path or ""))
    evidence["caller_model_revision"] = caller_revision
    expected_model_revision = lock.get("qwen_model_revision") if lock else None
    if not model_revision:
        reasons.append("model checkpoint has no revision attestation")
    elif expected_model_revision and model_revision != expected_model_revision:
        reasons.append(f"model checkpoint revision must equal {expected_model_revision}")
    if not caller_revision:
        reasons.append("caller checkpoint has no revision attestation")
    elif expected_model_revision and caller_revision != expected_model_revision:
        reasons.append(f"caller checkpoint revision must equal {expected_model_revision}")

    for label, value, expected_split in (
        ("rollout", experiment.rollout_dataset_path, "train"),
        ("validation", experiment.validation_dataset_path, "test"),
    ):
        result = _validate_dataset(Path(value or ""), expected_split, reasons)
        if result:
            evidence[f"{label}_dataset"] = result
    train = evidence.get("rollout_dataset", {})
    test = evidence.get("validation_dataset", {})
    overlap = sorted(set(train.get("task_ids", ())) & set(test.get("task_ids", ())))
    if overlap:
        reasons.append("training and held-out datasets contain overlapping task IDs")

    tau2_revision = _git_revision(Path(experiment.tau2_checkout or ""))
    evidence["tau2_revision"] = tau2_revision
    expected_tau2 = lock.get("tau2_commit") if lock else REQUIRED_LOCK_VALUES["tau2_commit"]
    if tau2_revision != expected_tau2:
        reasons.append(f"tau2 checkout must be pinned to {expected_tau2}")

    if probe_packages and lock:
        package_pins = {
            "verl-omni": lock.get("verl_omni_commit"),
            "verl": lock.get("verl_commit"),
            "vllm-omni": lock.get("vllm_omni_commit"),
        }
        installed: dict[str, Any] = {}
        for package, expected in package_pins.items():
            version, revision = _installed_distribution(package)
            installed[package] = {"version": version, "revision": revision}
            if revision != expected:
                reasons.append(f"installed {package} revision must equal {expected}")
        try:
            installed_audioenv = metadata.version("audioenv-foundry")
        except metadata.PackageNotFoundError:
            installed_audioenv = None
        installed["audioenv-foundry"] = {"version": installed_audioenv}
        if installed_audioenv != lock.get("audioenv_version"):
            reasons.append(f"installed AudioEnv version must equal {lock.get('audioenv_version')}")
        evidence["installed_packages"] = installed
        try:
            from ..training.verl_contract import validate_verl_runtime_contract
            evidence["verl_contract"] = validate_verl_runtime_contract()
        except Exception as exc:
            reasons.append(f"VeRL runtime contract failed: {exc}")

    if probe_cuda:
        _probe_cuda(reasons, evidence)
    return RuntimeAttestation(not reasons, tuple(reasons), evidence)


def _attest_python_environment(path: Path, reasons: list[str]) -> dict[str, Any]:
    """Bind the resolved build environment, not only top-level source pins."""

    if not path.is_file():
        reasons.append(f"resolved Python environment lock does not exist: {path}")
        return {}
    try:
        payload = path.read_bytes()
        lines = tuple(line for line in payload.decode("utf-8").splitlines() if line.strip())
    except (OSError, UnicodeDecodeError) as exc:
        reasons.append(f"resolved Python environment lock is unreadable: {exc}")
        return {}
    if not lines:
        reasons.append("resolved Python environment lock is empty")
        return {}
    if tuple(sorted(lines)) != lines:
        reasons.append("resolved Python environment lock must be deterministically sorted")
    return {
        "path": str(path.resolve()),
        "sha256": sha256(payload).hexdigest(),
        "distribution_count": len(lines),
    }


def _load_lock(path: Path, reasons: list[str]) -> dict[str, Any]:
    if not path.is_file():
        reasons.append(f"dependency lock does not exist: {path}")
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        reasons.append(f"dependency lock is unreadable: {exc}")
        return {}
    if not isinstance(value, dict):
        reasons.append("dependency lock must contain a JSON object")
        return {}
    return value


def _model_revision(path: Path) -> str | None:
    marker = path / "audioenv-model-revision.json"
    if marker.is_file():
        try:
            value = json.loads(marker.read_text(encoding="utf-8"))
            revision = value.get("revision")
            if isinstance(revision, str) and revision:
                return revision
        except (OSError, json.JSONDecodeError):
            return None
    config = path / "config.json"
    if config.is_file():
        try:
            revision = json.loads(config.read_text(encoding="utf-8")).get("_commit_hash")
            if isinstance(revision, str) and revision:
                return revision
        except (OSError, json.JSONDecodeError):
            pass
    if path.parent.name == "snapshots" and len(path.name) == 40:
        return path.name
    return None


def _validate_dataset(path: Path, expected_split: str, reasons: list[str]) -> dict[str, Any]:
    if not path.is_file():
        reasons.append(f"{expected_split} dataset does not exist: {path}")
        return {}
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    if not manifest_path.is_file():
        reasons.append(f"{expected_split} dataset manifest does not exist: {manifest_path}")
        return {}
    try:
        payload = path.read_bytes()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows = [json.loads(line) for line in payload.splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        reasons.append(f"{expected_split} dataset or manifest is unreadable: {exc}")
        return {}
    digest = sha256(payload).hexdigest()
    if manifest.get("dataset_sha256") != digest:
        reasons.append(f"{expected_split} dataset hash does not match its manifest")
    if manifest.get("split") != expected_split:
        reasons.append(f"{expected_split} dataset manifest has the wrong split")
    if manifest.get("row_count") != len(rows):
        reasons.append(f"{expected_split} dataset row count does not match its manifest")
    task_ids = tuple(sorted({str(row.get("task_id")) for row in rows}))
    if tuple(manifest.get("task_ids", ())) != task_ids:
        reasons.append(f"{expected_split} dataset task IDs do not match its manifest")
    if any(row.get("dataset_split") != expected_split for row in rows):
        reasons.append(f"{expected_split} dataset contains rows from another split")
    protected = {"private_goal", "expected_state", "oracle_actions", "verifier"}
    if any(protected & set(row) for row in rows):
        reasons.append(f"{expected_split} dataset exposes protected task material")
    return {
        "path": str(path.resolve()), "sha256": digest, "manifest_sha256": sha256(manifest_path.read_bytes()).hexdigest(),
        "split": manifest.get("split"), "row_count": len(rows), "task_ids": task_ids,
    }


def _git_revision(path: Path) -> str | None:
    if not path.is_dir():
        return None
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True, check=False,
    )
    return process.stdout.strip() if process.returncode == 0 else None


def _installed_distribution(package: str) -> tuple[str | None, str | None]:
    try:
        distribution = metadata.distribution(package)
    except metadata.PackageNotFoundError:
        return None, None
    direct_url = distribution.read_text("direct_url.json")
    if not direct_url:
        return distribution.version, None
    try:
        revision = json.loads(direct_url).get("vcs_info", {}).get("commit_id")
    except json.JSONDecodeError:
        revision = None
    return distribution.version, revision if isinstance(revision, str) else None


def _probe_cuda(reasons: list[str], evidence: dict[str, Any]) -> None:
    try:
        process = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=False,
        )
    except OSError as exc:
        reasons.append(f"nvidia-smi CUDA probe failed: {exc}")
        process = None
    if process is None:
        pass
    elif process.returncode:
        reasons.append("nvidia-smi CUDA probe failed")
    else:
        evidence["nvidia_smi"] = tuple(line.strip() for line in process.stdout.splitlines() if line.strip())
    try:
        import torch
        evidence["torch_version"] = torch.__version__
        evidence["torch_cuda_version"] = torch.version.cuda
        if torch.version.cuda is None or not torch.cuda.is_available():
            reasons.append("PyTorch CUDA runtime is unavailable")
    except Exception as exc:
        reasons.append(f"PyTorch CUDA probe failed: {exc}")
