"""Credential-free environment and model onboarding foundry.

Source inspection never imports or executes repository code. Recognized
declarative inputs can be compiled immediately; arbitrary repositories produce
a reviewable coding-environment scaffold and remain blocked until an OCI
sandbox is configured.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any

from .adapter import EnvironmentPackage
from .adapters import HarborEnvironmentAdapter, JsonBenchmarkAdapter
from .artifacts import LocalArtifactStore
from .authoring import AuthoredEnvironment, CustomTaskInput, DeterministicEnvironmentAuthor
from .contracts import (
    AuthoringStatus,
    EnvironmentPackageV1,
    EnvironmentType,
    ProtectedVerifierRef,
    PublicTaskV1,
    SourceProvenance,
)
from .harnesses import (
    HarnessRegistry,
    ModelOnboardingReport,
    SourceAnalysisReport,
    resolve_compatibility,
)
from .harnesses.codex_author import CodexEnvironmentAuthor
from .validation import ValidationReport, validate_package
from .models.scaffold import build_driver_scaffold
from .integrations.repo2rlenv import DockerSandboxExecutor, Repo2RLEnvGenerator


_GITHUB = re.compile(r"^https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?$")


@dataclass(frozen=True, slots=True)
class EnvironmentBuild:
    report: SourceAnalysisReport
    authored: AuthoredEnvironment | None = None
    generated_payload: dict[str, Any] | None = None
    boundary: EnvironmentPackageV1 | None = None


class SourceMaterializer:
    def __init__(self, workspace: Path):
        self.workspace = workspace.resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)

    def materialize(self, source_uri: str) -> tuple[Path, str | None]:
        local = Path(source_uri).expanduser()
        if local.exists():
            resolved = local.resolve()
            return resolved, _git_revision(resolved if resolved.is_dir() else resolved.parent)
        if not _GITHUB.fullmatch(source_uri):
            raise ValueError("source must be an existing local path or an https://github.com repository")
        destination = self.workspace / sha256(source_uri.encode()).hexdigest()[:16]
        if not destination.exists():
            process = subprocess.run(
                ["git", "clone", "--depth", "1", "--filter=blob:none", source_uri, str(destination)],
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            if process.returncode:
                raise ValueError(f"repository clone failed: {process.stderr.strip()[-500:]}")
        return destination, _git_revision(destination)


class EnvironmentFoundry:
    def __init__(
        self,
        workspace: Path,
        artifacts: LocalArtifactStore,
        codex_author: CodexEnvironmentAuthor | None = None,
        repo_generator: Repo2RLEnvGenerator | None = None,
    ):
        self.materializer = SourceMaterializer(workspace)
        self.artifacts = artifacts
        self.codex_author = codex_author or CodexEnvironmentAuthor()
        self.repo_generator = repo_generator or Repo2RLEnvGenerator(DockerSandboxExecutor())

    def build(
        self, source_uri: str, *, authoring_provider: str = "deterministic",
        repo_pipeline: str = "pr_diff", repo_limit: int = 5, repo_llm: str | None = None,
    ) -> EnvironmentBuild:
        source, revision = self.materializer.materialize(source_uri)
        root = source if source.is_dir() else source.parent
        inventory = _inventory(root)
        if authoring_provider == "repo2rlenv":
            if not source.is_dir():
                raise ValueError("Repo2RLEnv requires a repository directory")
            output = self.materializer.workspace / (sha256((source_uri + repo_pipeline).encode()).hexdigest()[:16] + "-harbor")
            output.mkdir(parents=True, exist_ok=True)
            generation = self.repo_generator.generate(
                source, output, pipeline=repo_pipeline, limit=repo_limit, llm=repo_llm,
            )
            boundary, bundles = HarborEnvironmentAdapter(self.artifacts).convert(
                output, environment_id=f"repo2rlenv/{source.name}",
                source_uri=source_uri, revision=revision,
            )
            report_artifact = self.artifacts.put_json(generation)
            report = SourceAnalysisReport(
                source_uri=source_uri, source_kind="repository", classification="harbor_repository",
                status="needs_review", detected_files=inventory, task_count=len(bundles),
                proposed_adapter="audioenv/harbor", verifier_source="Harbor private tests",
                execution_mode="oci_sandbox", generated_files=(report_artifact.id, boundary.protected_verifier.artifact_id),
                validation_checks=(), review_items=("run Harbor oracle, no-op, adversarial, determinism, leakage, and integrity gates",),
                environment_id=f"{boundary.id}@{boundary.version}",
            )
            return EnvironmentBuild(report, generated_payload=generation, boundary=boundary)
        task_manifest = _first_named(source, ("audioenv-task.json",))
        benchmark_manifest = _first_named(source, ("audioenv.json", "authoring.json", "benchmark.json"))
        if task_manifest is not None:
            raw = json.loads(task_manifest.read_text(encoding="utf-8"))
            authored = DeterministicEnvironmentAuthor(self.artifacts).author(CustomTaskInput.model_validate(raw))
            report = SourceAnalysisReport(
                source_uri=source_uri,
                source_kind="custom",
                classification="declarative_audio_task",
                status="available" if authored.validation.passed else "needs_review",
                detected_files=inventory,
                detected_tools=tuple(tool.name for tool in authored.package.spec.tools),
                task_count=len(authored.package.tasks),
                proposed_adapter="audioenv/deterministic-author",
                verifier_source="protected verifier generated from user-declared expected state",
                execution_mode="in_process",
                validation_checks=tuple(asdict(item) for item in authored.validation.checks),
                review_items=() if authored.validation.passed else ("environment validation failed",),
                environment_id=f"{authored.boundary.id}@{authored.boundary.version}",
            )
            return EnvironmentBuild(report, authored)
        if benchmark_manifest is not None:
            adapter = JsonBenchmarkAdapter({})
            summary = adapter.inspect(benchmark_manifest)
            raw = json.loads(benchmark_manifest.read_text(encoding="utf-8"))
            tool_names = tuple(tool["name"] for tool in raw["environment"].get("tools", []))
            if not tool_names:
                package = adapter.convert(benchmark_manifest)
                validation = validate_package(package)
                boundary = _boundary_for_package(
                    package, validation, self.artifacts, source_uri=source_uri, revision=revision,
                )
                authored = AuthoredEnvironment(package, boundary, validation)
                report = SourceAnalysisReport(
                    source_uri=source_uri, source_kind="benchmark", classification="audioenv_benchmark",
                    status="available" if validation.passed else "needs_review", detected_files=inventory,
                    task_count=summary["task_count"], proposed_adapter="audioenv/json-benchmark",
                    verifier_source=str(benchmark_manifest.relative_to(root)), execution_mode="in_process",
                    validation_checks=tuple(asdict(item) for item in validation.checks),
                    review_items=() if validation.passed else ("benchmark validation failed",),
                    environment_id=f"{boundary.id}@{boundary.version}",
                )
                return EnvironmentBuild(report, authored)
            report = SourceAnalysisReport(
                source_uri=source_uri, source_kind="benchmark", classification="audioenv_benchmark",
                status="needs_review", detected_files=inventory, detected_tools=tool_names,
                task_count=summary["task_count"], proposed_adapter="audioenv/json-benchmark",
                verifier_source=str(benchmark_manifest.relative_to(root)), execution_mode="not_executable",
                review_items=("bind audited tool handlers; repository code was not imported or executed",),
            )
            return EnvironmentBuild(report, generated_payload={"manifest": str(benchmark_manifest), "revision": revision})
        if authoring_provider == "codex":
            candidate = self.codex_author.author(
                root, source_uri=source_uri, inventory=inventory,
            )
            authored = DeterministicEnvironmentAuthor(self.artifacts).author(candidate)
            boundary_payload = authored.boundary.model_dump(mode="json")
            boundary_payload["authoring_status"] = AuthoringStatus.GENERATED_NEEDS_REVIEW
            boundary_payload.pop("content_hash", None)
            boundary_payload["content_hash"] = EnvironmentPackageV1.hash_payload(boundary_payload)
            boundary = EnvironmentPackageV1.model_validate(boundary_payload)
            authored = AuthoredEnvironment(authored.package, boundary, authored.validation)
            candidate_artifact = self.artifacts.put_json(candidate.model_dump(mode="json"))
            report = SourceAnalysisReport(
                source_uri=source_uri,
                source_kind="repository" if source.is_dir() else "custom",
                classification="codex_generated_candidate",
                status="needs_review",
                detected_files=inventory,
                detected_tools=tuple(tool.name for tool in candidate.tools),
                task_count=1,
                proposed_adapter="audioenv/codex-author+deterministic-author",
                verifier_source="generated candidate; protected verifier requires human approval",
                execution_mode="in_process",
                generated_files=(candidate_artifact.id,),
                validation_checks=tuple(asdict(item) for item in authored.validation.checks),
                review_items=(
                    "review the Codex-generated task, tools, and expected state",
                    "approve the protected verifier before changing authoring status to validated",
                ),
                environment_id=f"{boundary.id}@{boundary.version}",
            )
            return EnvironmentBuild(report, authored, candidate.model_dump(mode="json"))
        harbor_manifest = _harbor_manifest(source)
        if harbor_manifest is not None:
            boundary, bundles = HarborEnvironmentAdapter(self.artifacts).convert(
                root, environment_id=f"harbor/{root.name}", source_uri=source_uri, revision=revision,
            )
            report = SourceAnalysisReport(
                source_uri=source_uri, source_kind="repository", classification="harbor_repository",
                status="needs_review", detected_files=inventory, task_count=len(bundles),
                proposed_adapter="audioenv/harbor", verifier_source="Harbor private tests",
                execution_mode="oci_sandbox", generated_files=(boundary.protected_verifier.artifact_id,),
                review_items=("run the protected sandbox promotion gates",),
                environment_id=f"{boundary.id}@{boundary.version}",
            )
            return EnvironmentBuild(report, boundary=boundary)
        if authoring_provider != "deterministic":
            raise ValueError("authoring_provider must be deterministic, codex, or repo2rlenv")
        tests = tuple(item for item in inventory if _looks_like_test(item))
        if tests:
            scaffold = {
                "schema_version": "audioenv.repository-scaffold/v1",
                "source_uri": source_uri,
                "revision": revision,
                "environment_type": "coding",
                "detected_tests": tests[:100],
                "sandbox": {
                    "network": "off",
                    "run_as_non_root": True,
                    "read_only_base": True,
                    "ephemeral_workspace": True,
                    "status": "required_not_configured",
                },
                "verifier": {"kind": "repository_tests", "protected": True},
            }
            artifact = self.artifacts.put_json(scaffold)
            report = SourceAnalysisReport(
                source_uri=source_uri, source_kind="repository", classification="coding_repository",
                status="needs_review", detected_files=inventory, task_count=len(tests),
                proposed_adapter="audioenv/repository-coding", verifier_source="detected repository tests",
                execution_mode="oci_sandbox", generated_files=(artifact.id,),
                review_items=(
                    "confirm the test command and task-generation strategy",
                    "configure and validate an OCI sandbox before executing repository code",
                ),
            )
            return EnvironmentBuild(report, generated_payload=scaffold)
        return EnvironmentBuild(SourceAnalysisReport(
            source_uri=source_uri, source_kind="unknown", classification="unsupported_source",
            status="unsupported", detected_files=inventory,
            review_items=("no AudioEnv manifest or supported repository test surface was detected",),
        ))


class ModelFoundry:
    def __init__(self, registry: HarnessRegistry, artifacts: LocalArtifactStore):
        self.registry = registry
        self.artifacts = artifacts

    def inspect(self, model_id: str, source: str) -> ModelOnboardingReport:
        config, checks = _read_model_config(source)
        family = _detect_model_family(model_id, config)
        if family == "qwen3-omni":
            driver = "audioenv/qwen3-omni"
            capabilities = self.registry.get(driver).capabilities  # type: ignore[union-attr]
            compatibility = resolve_compatibility(
                self.registry, environment_id="unselected", environment_requirements=("audio-input",),
                model_id=model_id, model_capabilities=capabilities,
            )
            return ModelOnboardingReport(
                model_id=model_id, source=source, detected_family=family, selected_driver=driver,
                status="blocked", registration_state="generated_needs_review", capabilities=capabilities,
                conformance_checks=tuple(checks),
                compatible_trainers=tuple(
                    item.harness_id for item in compatibility.trainers
                    if not item.reasons or item.harness_id.startswith("qwen3-omni-")
                ),
                review_items=("run the CUDA load, native-audio generation, log-probability, LoRA, and reload conformance gates",),
            )
        artifact = self.artifacts.put_bytes(
            build_driver_scaffold(model_id, source, family), media_type="application/zip",
        )
        return ModelOnboardingReport(
            model_id=model_id, source=source, detected_family=family, selected_driver=None,
            status="needs_review", registration_state="generated_needs_review",
            conformance_checks=tuple(checks), generated_files=(artifact.id,),
            review_items=("implement the generated architecture-family driver and pass its conformance suite",),
        )


def _boundary_for_package(
    package: EnvironmentPackage,
    validation: ValidationReport,
    artifacts: LocalArtifactStore,
    *,
    source_uri: str,
    revision: str | None,
) -> EnvironmentPackageV1:
    verifier = artifacts.put_json({
        "tasks": [{
            "id": task.id,
            "expected_state": task.expected_state,
            "required_tools": [asdict(item) for item in task.required_tools],
            "required_communications": task.required_communications,
            "forbidden_tools": task.forbidden_tools,
        } for task in package.tasks],
    })
    payload: dict[str, Any] = {
        "schema_version": "audioenv.environment/v1",
        "id": package.spec.id,
        "version": package.spec.version,
        "description": package.spec.description,
        "environment_type": EnvironmentType.CONVERSATIONAL,
        "source": SourceProvenance(kind="benchmark", uri=source_uri, revision=revision).model_dump(mode="json"),
        "required_capabilities": package.spec.model_requirements,
        "tasks": [PublicTaskV1(
            id=task.id, split=task.split, goal=task.user_goal,
            tool_names=tuple(tool.name for tool in package.spec.tools),
            audio_profile_ids=tuple(profile.id for profile in package.spec.audio_profiles),
        ).model_dump(mode="json") for task in package.tasks],
        "protected_verifier": ProtectedVerifierRef(artifact_id=verifier.id, sha256=verifier.sha256).model_dump(mode="json"),
        "authoring_status": AuthoringStatus.VALIDATED if validation.passed else AuthoringStatus.GENERATED_NEEDS_REVIEW,
        "parent_hash": None,
    }
    payload["content_hash"] = EnvironmentPackageV1.hash_payload(payload)
    return EnvironmentPackageV1.model_validate(payload)


def _inventory(root: Path, limit: int = 500) -> tuple[str, ...]:
    items: list[str] = []
    for path in sorted(root.rglob("*")):
        if len(items) >= limit:
            break
        if path.is_symlink() or not path.is_file():
            continue
        if ".git" in path.parts or any(part in {"node_modules", ".venv", "dist", "build"} for part in path.parts):
            continue
        try:
            if path.stat().st_size > 2_000_000:
                continue
        except OSError:
            continue
        items.append(path.relative_to(root).as_posix())
    return tuple(items)


def _first_named(source: Path, names: tuple[str, ...]) -> Path | None:
    if source.is_file() and source.name in names:
        return source
    if source.is_dir():
        for name in names:
            direct = source / name
            if direct.is_file():
                return direct
        for path in source.rglob("*.json"):
            if path.name in names and ".git" not in path.parts:
                return path
    return None


def _harbor_manifest(source: Path) -> Path | None:
    direct = _first_named(source, ("task.toml", "registry.json"))
    if direct is not None or not source.is_dir():
        return direct
    return next(
        (path for path in sorted(source.glob("*/task.toml")) if ".git" not in path.parts),
        None,
    )


def _looks_like_test(path: str) -> bool:
    name = Path(path).name
    return name.startswith("test_") or name.endswith("_test.py") or name.endswith(".test.ts") or name.endswith(".spec.ts")


def _git_revision(path: Path) -> str | None:
    if shutil.which("git") is None:
        return None
    process = subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True, check=False)
    return process.stdout.strip() if process.returncode == 0 else None


def _read_model_config(source: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = Path(source).expanduser()
    config_path = path / "config.json" if path.is_dir() else path
    checks: list[dict[str, Any]] = []
    if config_path.is_file() and config_path.name == "config.json":
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            checks.append({"name": "config_json", "passed": True, "details": str(config_path)})
            return config, checks
        except (OSError, json.JSONDecodeError) as exc:
            checks.append({"name": "config_json", "passed": False, "details": str(exc)})
    else:
        checks.append({"name": "config_json", "passed": False, "details": "remote identifiers are inspected by name until downloaded"})
    return {}, checks


def _detect_model_family(model_id: str, config: dict[str, Any]) -> str:
    values = [model_id, str(config.get("model_type", "")), *(str(item) for item in config.get("architectures", []))]
    normalized = " ".join(values).casefold().replace("_", "-")
    if "qwen3-omni" in normalized or "qwen3omni" in normalized:
        return "qwen3-omni"
    if "qwen2.5-omni" in normalized or "qwen2-5-omni" in normalized:
        return "qwen2.5-omni"
    return "unknown"
