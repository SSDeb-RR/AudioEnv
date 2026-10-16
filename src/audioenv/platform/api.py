"""FastAPI control plane and production SPA host for AudioEnv."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import asdict
from datetime import UTC, datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Callable, Literal

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, StreamingResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel, ConfigDict, Field
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Install AudioEnv with the 'platform' extra") from exc

from .db import Database, InvalidTransition, PlatformRepository, serialize_experiment
from ..artifacts import LocalArtifactStore
from ..authoring import CustomTaskInput, DeterministicEnvironmentAuthor
from ..evaluation import assess_learning_evidence, build_evaluation_result, package_hashes
from ..contracts import (
    EnvironmentPackageV1, GPUExecutionPlanV1, GPUWorkerExperimentV1,
    EvaluationProtocolV1, EvaluationSampleV1, PromotionEvidenceV1, SandboxPolicyV1,
)
from ..foundry import EnvironmentFoundry, ModelFoundry
from ..harnesses import build_builtin_registry, resolve_compatibility
from ..isolated_validation import IsolatedValidationError, IsolatedValidationRunner
from ..reference import build_reference_package, build_reference_split_manifest
from ..promotion import PromotionEvidenceAuthority
from ..training import TinyAudioPolicyTrainer, TinyTrainerConfig
from ..validation import validate_package


class StrictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProjectCreate(StrictRequest):
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=2_000)


class ExperimentCreate(StrictRequest):
    name: str = Field(min_length=1, max_length=200)
    project_id: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)


class ExperimentResume(StrictRequest):
    checkpoint_artifact_id: str = Field(min_length=64, max_length=64)
    run_name: str = Field(min_length=1, max_length=200)
    additional_steps: int = Field(default=10, ge=1)


class TransitionRequest(StrictRequest):
    target: str
    payload: dict[str, Any] = Field(default_factory=dict)


class MetricCreate(StrictRequest):
    step: int = Field(ge=0)
    name: str = Field(min_length=1)
    value: float
    source: str = Field(min_length=1)


class ModelCreate(StrictRequest):
    id: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    adapter_type: str = Field(min_length=1)
    role: str = Field(pattern="^(agent|caller|both)$")
    project_id: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)


class LaunchRequest(StrictRequest):
    confirmed: bool
    provider: str = "local-subprocess"
    confirmation_name: str | None = None
    quote_id: str | None = None
    compute_resource_id: str | None = None


class RunPodQuoteRequest(StrictRequest):
    image: str = Field(min_length=1, max_length=500)
    image_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    gpu_count: int = Field(default=8, ge=1)
    gpu_memory_gb: int = Field(default=80, ge=1)
    max_runtime_hours: float = Field(gt=0)
    max_total_budget_usd: float = Field(ge=0)
    network_volume_id: str | None = Field(default=None, max_length=500)
    ssh_key_secret_name: str = Field(default="RUNPOD_SSH_KEY", pattern=r"^[A-Z][A-Z0-9_]{0,127}$")
    api_key_secret_name: str = Field(default="RUNPOD_API_KEY", pattern=r"^[A-Z][A-Z0-9_]{0,127}$")
    known_hosts_secret_name: str = Field(default="RUNPOD_KNOWN_HOSTS", pattern=r"^[A-Z][A-Z0-9_]{0,127}$")


class ModalQuoteRequest(StrictRequest):
    image: str = Field(min_length=1, max_length=500)
    image_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    gpu_type: str = Field(default="H100", min_length=1, max_length=100)
    gpu_count: int = Field(default=8, ge=1)
    max_runtime_hours: float = Field(gt=0, le=24)
    max_total_budget_usd: float = Field(ge=0)
    volume_name: str = Field(default="audioenv-runs", min_length=1, max_length=200)
    secret_names: tuple[str, ...] = ()
    hourly_price_per_gpu_usd: float = Field(gt=0)


class LocalCUDAPrepareRequest(StrictRequest):
    plan: GPUExecutionPlanV1
    data_dir: str = Field(min_length=1, max_length=4_096)
    model_path: str = Field(min_length=1, max_length=4_096)


class ModelProbeRequest(StrictRequest):
    model_id: str


class ModelConformanceSandboxRequest(StrictRequest):
    image: str = Field(pattern=r"^(?:[^\s@]+@)?sha256:[0-9a-f]{64}$")
    checkpoint_path: str | None = Field(default=None, max_length=4_096)


class ModelConformanceImageBuildRequest(StrictRequest):
    base_image: str = Field(pattern=r"^(?:[^\s@]+@)?sha256:[0-9a-f]{64}$")


class GPUPreflightRequest(StrictRequest):
    plan: GPUExecutionPlanV1
    model_path: str


class EnvironmentFoundryRequest(StrictRequest):
    project_id: str | None = None
    source_uri: str = Field(min_length=1, max_length=4_096)
    authoring_provider: Literal["deterministic", "codex", "repo2rlenv"] = "deterministic"
    repo_pipeline: Literal["pr_runtime", "commit_runtime", "pr_diff"] = "pr_diff"
    repo_limit: int = Field(default=5, ge=1, le=100)
    repo_llm: str | None = Field(default=None, max_length=200)


class EnvironmentPromotionRequest(StrictRequest):
    evidence_artifact_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    confirmation_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class HarborValidationRequest(StrictRequest):
    source_uri: str | None = Field(default=None, max_length=4_096)
    policy: SandboxPolicyV1 = Field(default_factory=SandboxPolicyV1)


class ModelFoundryRequest(StrictRequest):
    project_id: str | None = None
    model_id: str = Field(min_length=1, max_length=500)
    source: str = Field(min_length=1, max_length=4_096)
    role: str = Field(default="agent", pattern="^(agent|caller|both)$")


class CompatibilityRequest(StrictRequest):
    environment_id: str
    model_id: str
    caller_id: str | None = None


class LearningEvidenceRequest(StrictRequest):
    baseline_evaluation_id: str
    trained_evaluation_id: str
    independent_baseline_evaluation_id: str | None = None
    independent_trained_evaluation_id: str | None = None
    reward_parity_artifact_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


def create_app(
    database_url: str | None = None, *, static_dir: Path | None = None,
    runpod_transport: Any | None = None,
    runpod_executor_factory: Callable[[dict[str, Any], Any], Any] | None = None,
    modal_transport: Any | None = None,
    local_cuda_provider: Any | None = None,
    harbor_transport: Any | None = None,
    model_conformance_transport: Any | None = None,
    model_conformance_image_builder: Any | None = None,
    trusted_model_conformance_images: tuple[str, ...] | None = None,
    external_resource_poll_interval_seconds: float | None = None,
) -> FastAPI:
    resolved_url = database_url or os.environ.get("AUDIOENV_DATABASE_URL", "sqlite:///./.audioenv/audioenv.db")
    database = Database(resolved_url)
    database.create_schema()
    repository = PlatformRepository(database)
    repository.ensure_default_project()
    app = FastAPI(title="AudioEnv", version="0.5.0")
    app.state.database = database
    app.state.repository = repository
    app.state.processes = {}
    if resolved_url.startswith("sqlite:///"):
        data_root = Path(resolved_url.removeprefix("sqlite:///")).resolve().parent
        artifact_root = data_root / "artifacts"
    else:
        data_root = Path(os.environ.get("AUDIOENV_DATA_ROOT", ".audioenv")).resolve()
        artifact_root = Path(os.environ.get("AUDIOENV_ARTIFACT_ROOT", data_root / "artifacts")).resolve()
    app.state.data_root = data_root
    app.state.artifacts = LocalArtifactStore(artifact_root)
    app.state.model_conformance_transport = model_conformance_transport
    app.state.model_conformance_image_builder = model_conformance_image_builder
    configured_images = os.environ.get("AUDIOENV_MODEL_CONFORMANCE_IMAGES", "")
    app.state.trusted_model_conformance_images = set(
        trusted_model_conformance_images
        if trusted_model_conformance_images is not None
        else (item.strip() for item in configured_images.split(",") if item.strip())
    )
    from ..integrations.model_conformance_sandbox import ImageBuildEvidenceAuthority
    app.state.image_build_authority = ImageBuildEvidenceAuthority(
        data_root / "secrets" / "model-image-evidence.key",
    )
    _restore_trusted_model_images(app, repository)
    app.state.promotion_authority = PromotionEvidenceAuthority(data_root / "secrets" / "promotion-evidence.key")
    app.state.harness_registry = build_builtin_registry()
    app.state.environment_foundry = EnvironmentFoundry(data_root / "foundry" / "sources", app.state.artifacts)
    app.state.model_foundry = ModelFoundry(app.state.harness_registry, app.state.artifacts)
    app.state.validation_runner = IsolatedValidationRunner()
    app.state.runpod_transport = runpod_transport
    app.state.runpod_executor_factory = runpod_executor_factory
    app.state.modal_transport = modal_transport
    app.state.local_cuda_provider = local_cuda_provider
    app.state.harbor_transport = harbor_transport
    app.state.external_resource_poll_interval_seconds = external_resource_poll_interval_seconds
    app.state.external_resource_watchdog_task = None
    app.state.external_resource_watchdog_errors = {}
    _seed_builtin_records(repository)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_methods=["GET", "POST"],
        allow_headers=["content-type"],
    )

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": "0.5.0"}

    @app.get("/api/overview")
    def overview() -> dict[str, Any]:
        return _redact_public(repository.overview())

    @app.post("/api/projects", status_code=201)
    def create_project(request: ProjectCreate) -> dict[str, Any]:
        try:
            row = repository.create_project(request.name, request.description)
        except Exception as exc:
            raise HTTPException(409, "project name already exists") from exc
        return _project(row)

    @app.get("/api/projects")
    def projects() -> list[dict[str, Any]]:
        return [_project(row) for row in repository.list_projects()]

    @app.get("/api/projects/{project_id}")
    def project(project_id: str) -> dict[str, Any]:
        row = repository.get_project(project_id)
        if row is None:
            raise HTTPException(404, "project not found")
        return _project(row)

    @app.get("/api/trainers")
    def trainers() -> list[dict[str, Any]]:
        return [item.model_dump(mode="json") for item in app.state.harness_registry.list("trainer")]

    @app.get("/api/providers")
    def providers() -> list[dict[str, Any]]:
        rows = []
        for item in app.state.harness_registry.list("provider"):
            row = item.model_dump(mode="json")
            row.update({
                "supports_quote": item.id in {"runpod-secure-cloud", "modal-sandbox"},
                "requires_quote": item.id in {"runpod-secure-cloud", "modal-sandbox"},
                "requires_machine_preflight": item.id == "local-cuda",
                "estimated_cost_usd": 0.0 if item.id == "local-subprocess" else None,
            })
            rows.append(row)
        return rows

    @app.get("/api/harnesses")
    def harnesses(kind: str | None = None) -> list[dict[str, Any]]:
        valid = {"environment", "model", "caller", "trainer", "provider"}
        if kind is not None and kind not in valid:
            raise HTTPException(422, f"kind must be one of {sorted(valid)}")
        return [item.model_dump(mode="json") for item in app.state.harness_registry.list(kind)]

    @app.get("/api/foundry/builds")
    def foundry_builds(kind: str | None = None) -> list[dict[str, Any]]:
        return [_harness_build(row) for row in repository.list_harness_builds(kind)]

    @app.get("/api/foundry/builds/{build_id}")
    def foundry_build(build_id: str) -> dict[str, Any]:
        row = repository.get_harness_build(build_id)
        if row is None:
            raise HTTPException(404, "harness build not found")
        return _harness_build(row)

    @app.post("/api/foundry/environments", status_code=201)
    def build_environment(request: EnvironmentFoundryRequest) -> dict[str, Any]:
        if request.project_id and repository.get_project(request.project_id) is None:
            raise HTTPException(422, "project not found")
        try:
            result = app.state.environment_foundry.build(
                request.source_uri, authoring_provider=request.authoring_provider,
                repo_pipeline=request.repo_pipeline, repo_limit=request.repo_limit,
                repo_llm=request.repo_llm,
            )
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            raise HTTPException(422, str(exc)) from exc
        environment_id = None
        boundary = result.authored.boundary if result.authored is not None else result.boundary
        if boundary is not None:
            validation = {
                "passed": result.authored.validation.passed if result.authored else False,
                "checks": [asdict(item) for item in result.authored.validation.checks] if result.authored else [],
            }
            row = repository.register_environment(
                record_id=f"{boundary.id}@{boundary.version}", version=boundary.version,
                content_hash=boundary.content_hash, source_kind=boundary.source.kind,
                authoring_status=boundary.authoring_status, package=boundary.model_dump(mode="json"),
                validation=validation, project_id=request.project_id,
            )
            environment_id = row.id
        report = result.report.model_dump(mode="json")
        _register_foundry_artifacts(
            repository, app.state.artifacts, result.report.generated_files,
            project_id=request.project_id, metadata={
                "kind": "environment", "source_uri": request.source_uri,
                "authoring_provider": request.authoring_provider,
            },
        )
        build = repository.add_harness_build(
            project_id=request.project_id, kind="environment", source_uri=request.source_uri,
            status=result.report.status, report=report,
            generated_artifact_ids=result.report.generated_files, environment_id=environment_id,
        )
        return _harness_build(build)

    @app.post("/api/environments/{environment_id:path}/promotion-evidence", status_code=201)
    def create_promotion_evidence(environment_id: str) -> dict[str, Any]:
        row = repository.get_environment(environment_id)
        if row is None:
            raise HTTPException(404, "environment not found")
        package = json.loads(row.package_json)
        verifier = package.get("protected_verifier")
        if not isinstance(verifier, dict):
            raise HTTPException(422, "environment has no protected verifier reference")
        try:
            verifier_payload = app.state.artifacts.get_bytes(str(verifier["artifact_id"]))
        except (KeyError, FileNotFoundError, ValueError) as exc:
            raise HTTPException(422, "protected verifier artifact is unavailable") from exc
        actual_verifier_hash = sha256(verifier_payload).hexdigest()
        if actual_verifier_hash != verifier.get("sha256"):
            raise HTTPException(422, "protected verifier artifact hash mismatch")
        validation = json.loads(row.validation_json) if row.validation_json else {}
        evidence = app.state.promotion_authority.issue(
            environment_hash=row.content_hash,
            verifier_hash_before=actual_verifier_hash,
            verifier_hash_after=actual_verifier_hash,
            validation=validation,
        )
        stored = app.state.artifacts.put_json(evidence.model_dump(mode="json"))
        repository.add_artifact(
            stored.id, experiment_id=None, project_id=row.project_id,
            kind="promotion-evidence", media_type=stored.media_type,
            size_bytes=stored.size_bytes, path=str(stored.path),
            metadata={"environment_id": row.id, "environment_hash": row.content_hash, "worker_id": evidence.worker_id},
        )
        report = app.state.promotion_authority.verify(evidence)
        return {"artifact_id": stored.id, "evidence": evidence.model_dump(mode="json"), "report": report.model_dump(mode="json")}

    @app.post("/api/environments/{environment_id:path}/promote")
    def promote_environment(environment_id: str, request: EnvironmentPromotionRequest) -> dict[str, Any]:
        row = repository.get_environment(environment_id)
        if row is None:
            raise HTTPException(404, "environment not found")
        if request.confirmation_hash != row.content_hash:
            raise HTTPException(409, "confirmation hash does not match the current environment")
        try:
            payload = json.loads(app.state.artifacts.get_bytes(request.evidence_artifact_id))
            evidence = PromotionEvidenceV1.model_validate(payload)
            report = app.state.promotion_authority.verify(evidence)
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
            raise HTTPException(422, f"invalid promotion evidence: {exc}") from exc
        if evidence.environment_hash != row.content_hash:
            raise HTTPException(409, "promotion evidence belongs to a different environment version")
        if not report.promoted:
            raise HTTPException(422, {"message": "promotion gates failed", "report": report.model_dump(mode="json")})
        final_report = report.model_copy(update={"evidence_artifact_id": request.evidence_artifact_id})
        promoted = repository.promote_environment(environment_id, final_report.model_dump(mode="json"))
        return _environment(promoted, details=True)

    @app.post("/api/foundry/models", status_code=201)
    def build_model_harness(request: ModelFoundryRequest) -> dict[str, Any]:
        if request.project_id and repository.get_project(request.project_id) is None:
            raise HTTPException(422, "project not found")
        report = app.state.model_foundry.inspect(request.model_id, request.source)
        adapter_type = report.selected_driver.removeprefix("audioenv/") if report.selected_driver else "generated-needs-review"
        repository.register_model(
            request.model_id, request.model_id.rsplit("/", 1)[-1], adapter_type, request.role,
            {"checkpoint": request.source, "onboarding": report.model_dump(mode="json")},
            request.project_id, report.registration_state,
        )
        _register_foundry_artifacts(
            repository, app.state.artifacts, report.generated_files,
            project_id=request.project_id, metadata={"kind": "model", "model_id": request.model_id},
        )
        build = repository.add_harness_build(
            project_id=request.project_id, kind="model", source_uri=request.source,
            status=report.status, report=report.model_dump(mode="json"),
            generated_artifact_ids=report.generated_files, model_id=request.model_id,
        )
        return _harness_build(build)

    @app.post("/api/compatibility")
    def compatibility(request: CompatibilityRequest) -> dict[str, Any]:
        environment = repository.get_environment(request.environment_id)
        model = repository.get_model(request.model_id)
        if environment is None:
            raise HTTPException(404, "environment not found")
        if model is None:
            raise HTTPException(404, "model not found")
        package = json.loads(environment.package_json)
        requirements = tuple(package.get("required_capabilities", ("audio-input",)))
        probe = json.loads(model.probe_json) if model.probe_json else {}
        capability_map = probe.get("capabilities") or json.loads(model.config_json).get("onboarding", {}).get("capabilities", [])
        if isinstance(capability_map, dict):
            capabilities = tuple(_capability_names(capability_map))
        else:
            capabilities = tuple(capability_map)
        return resolve_compatibility(
            app.state.harness_registry, environment_id=environment.id,
            environment_requirements=requirements, model_id=model.id,
            model_capabilities=capabilities, caller_id=request.caller_id,
        ).model_dump(mode="json")

    @app.post("/api/models", status_code=201)
    def register_model(request: ModelCreate) -> dict[str, Any]:
        try:
            _assert_no_embedded_secrets(request.config)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        row = repository.register_model(
            request.id, request.display_name, request.adapter_type, request.role,
            request.config, request.project_id,
        )
        return _model(row)

    @app.get("/api/models")
    def models() -> list[dict[str, Any]]:
        return [_model(row) for row in repository.list_models()]

    @app.post("/api/models/probe")
    def probe_model(request: ModelProbeRequest) -> dict[str, Any]:
        row = repository.get_model(request.model_id)
        if row is None:
            raise HTTPException(404, "model not found")
        config = json.loads(row.config_json)
        if row.adapter_type == "qwen3-omni":
            from ..models import Qwen3OmniAgentAdapter
            probe = Qwen3OmniAgentAdapter(
                config.get("checkpoint", row.id), revision=config.get("revision"), allow_download=False,
            ).probe()
        elif row.adapter_type == "small-audio-tool-policy":
            trainer = TinyAudioPolicyTrainer(build_reference_package())
            estimate = trainer.preflight(TinyTrainerConfig(steps=1))
            probe = {
                "available": estimate.available,
                "model_id": row.id,
                "device": "cpu",
                "revision": "builtin-v2",
                "reasons": estimate.reasons,
                "capabilities": {
                    "audio_input": True, "audio_output": False, "structured_tools": True,
                    "token_logprobs": True, "trainable": True, "batching": False,
                    "distributed": False, "trainable_modules": ["audio_candidate_scorer"],
                    "checkpoint_format": "pytorch-state-dict",
                },
            }
            updated = repository.set_model_probe(request.model_id, probe)
            return _model(updated)
        else:
            raise HTTPException(422, f"unsupported model adapter: {row.adapter_type}")
        updated = repository.set_model_probe(request.model_id, probe.model_dump(mode="json"))
        return _model(updated)

    @app.post("/api/models/{model_id:path}/conformance")
    def model_conformance(model_id: str) -> dict[str, Any]:
        row = repository.get_model(model_id)
        if row is None:
            raise HTTPException(404, "model not found")
        if row.adapter_type != "qwen3-omni":
            raise HTTPException(422, "full conformance is currently implemented for qwen3-omni")
        from ..audio import DeterministicWavFixtureRenderer
        from ..models import (
            ConformanceFixture, ModelConformanceRunner, Qwen3OmniAgentAdapter, Qwen3OmniDriver,
        )
        from ..spec import AudioObservation, AudioProfile
        config = json.loads(row.config_json)
        adapter = Qwen3OmniAgentAdapter(
            config.get("checkpoint", row.id), revision=config.get("revision"), allow_download=False,
        )
        audio = DeterministicWavFixtureRenderer().render("Please check my order.", AudioProfile(), 7)
        report = ModelConformanceRunner().run(
            Qwen3OmniDriver(adapter), ConformanceFixture(AudioObservation(audio=audio)),
        )
        probe = adapter.probe().model_dump(mode="json")
        probe["available"] = report.registered
        probe["registration_state"] = report.registration_state
        probe["conformance"] = report.model_dump(mode="json")
        repository.set_model_probe(model_id, probe)
        return report.model_dump(mode="json")

    @app.post("/api/models/{model_id:path}/driver-package", status_code=201)
    async def upload_model_driver_package(model_id: str, request: Request) -> dict[str, Any]:
        row = repository.get_model(model_id)
        if row is None:
            raise HTTPException(404, "model not found")
        if row.registration_state == "certified":
            raise HTTPException(409, "certified built-in model drivers cannot be replaced")
        if request.headers.get("content-type", "").split(";", 1)[0] not in {
            "application/zip", "application/octet-stream",
        }:
            raise HTTPException(415, "driver package must be uploaded as application/zip")
        try:
            payload = await _read_limited_body(request, 4 * 1024 * 1024)
        except ValueError as exc:
            raise HTTPException(413, str(exc)) from exc
        try:
            from ..models.package import inspect_driver_package
            inspection = inspect_driver_package(payload, expected_model_id=model_id)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        stored = app.state.artifacts.put_bytes(payload, media_type="application/zip")
        repository.add_artifact(
            stored.id, experiment_id=None, project_id=row.project_id,
            kind="model-driver-package", media_type=stored.media_type,
            size_bytes=stored.size_bytes, path=str(stored.path), metadata={
                "model_id": model_id, "package_sha256": inspection.sha256,
                "registration_state": "generated_needs_review",
            },
        )
        updated = repository.set_model_driver_package(model_id, {
            "artifact_id": stored.id, "sha256": inspection.sha256,
            "manifest": inspection.manifest, "files": list(inspection.files),
            "uncompressed_bytes": inspection.uncompressed_bytes,
        })
        return {"model": _model(updated), "artifact_id": stored.id,
                "inspection": asdict(inspection)}

    @app.post("/api/models/{model_id:path}/conformance-sandbox")
    def run_model_conformance_sandbox(
        model_id: str, request: ModelConformanceSandboxRequest,
    ) -> dict[str, Any]:
        row = repository.get_model(model_id)
        if row is None:
            raise HTTPException(404, "model not found")
        if row.registration_state == "certified":
            raise HTTPException(409, "certified built-in drivers do not use uploaded conformance")
        if request.image not in app.state.trusted_model_conformance_images:
            raise HTTPException(
                422, "conformance image is not in the operator-approved immutable image set",
            )
        config = json.loads(row.config_json)
        package = config.get("driver_package")
        if not isinstance(package, dict) or not package.get("artifact_id"):
            raise HTTPException(422, "upload an implemented driver package before conformance")
        artifact = repository.get_artifact(str(package["artifact_id"]))
        if artifact is None or artifact.kind != "model-driver-package":
            raise HTTPException(422, "registered driver package artifact is unavailable")
        metadata = json.loads(artifact.metadata_json)
        if metadata.get("model_id") != model_id or metadata.get("package_sha256") != artifact.id:
            raise HTTPException(422, "driver package artifact binding is invalid")
        try:
            package_payload = app.state.artifacts.get_bytes(artifact.id)
        except (OSError, ValueError) as exc:
            raise HTTPException(422, "driver package artifact is unavailable") from exc
        if sha256(package_payload).hexdigest() != artifact.id:
            raise HTTPException(422, "driver package artifact failed content-hash verification")
        package_path = app.state.artifacts.describe(
            artifact.id, media_type="application/zip",
        ).path
        try:
            from ..integrations.model_conformance_sandbox import (
                DockerModelConformanceTransport, ModelConformanceSandbox,
            )
            transport = app.state.model_conformance_transport or DockerModelConformanceTransport()
            report = ModelConformanceSandbox(transport).validate(
                package_path=package_path, model_id=model_id, image=request.image,
                checkpoint_path=Path(request.checkpoint_path) if request.checkpoint_path else None,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise HTTPException(422, f"isolated model conformance failed: {exc}") from exc
        stored = app.state.artifacts.put_json(report.model_dump(mode="json"))
        repository.add_artifact(
            stored.id, experiment_id=None, project_id=row.project_id,
            kind="model-conformance-report", media_type=stored.media_type,
            size_bytes=stored.size_bytes, path=str(stored.path), metadata={
                "model_id": model_id, "driver_package_artifact_id": artifact.id,
                "runtime_image": request.image, "registered": report.registered,
            },
        )
        probe = {
            "available": report.registered,
            "registration_state": report.registration_state,
            "conformance": report.model_dump(mode="json"),
            "conformance_artifact_id": stored.id,
            "runtime_image": request.image,
        }
        updated = repository.set_model_probe(model_id, probe)
        return {"model": _model(updated), "report": report.model_dump(mode="json"),
                "artifact_id": stored.id}

    @app.post("/api/models/{model_id:path}/conformance-image", status_code=201)
    def build_model_conformance_image(
        model_id: str, request: ModelConformanceImageBuildRequest,
    ) -> dict[str, Any]:
        row = repository.get_model(model_id)
        if row is None:
            raise HTTPException(404, "model not found")
        if request.base_image not in app.state.trusted_model_conformance_images:
            raise HTTPException(422, "base image is not operator-approved")
        config = json.loads(row.config_json)
        package = config.get("driver_package")
        if not isinstance(package, dict) or not package.get("artifact_id"):
            raise HTTPException(422, "upload an implemented driver package before building")
        artifact_id = str(package["artifact_id"])
        try:
            payload = app.state.artifacts.get_bytes(artifact_id)
        except (OSError, ValueError) as exc:
            raise HTTPException(422, "driver package artifact is unavailable") from exc
        if sha256(payload).hexdigest() != artifact_id:
            raise HTTPException(422, "driver package artifact failed content-hash verification")
        package_path = app.state.artifacts.describe(
            artifact_id, media_type="application/zip",
        ).path
        try:
            from ..integrations.model_conformance_sandbox import (
                DockerModelConformanceImageBuilder,
            )
            builder = app.state.model_conformance_image_builder or DockerModelConformanceImageBuilder()
            receipt = builder.build(package_path=package_path, base_image=request.base_image)
        except (OSError, RuntimeError, ValueError) as exc:
            raise HTTPException(422, f"conformance image build failed: {exc}") from exc
        if receipt.get("driver_package_sha256") != artifact_id:
            raise HTTPException(422, "conformance image receipt is bound to a different driver package")
        image = str(receipt.get("image", ""))
        if not re.fullmatch(r"(?:[^\s@]+@)?sha256:[0-9a-f]{64}", image):
            raise HTTPException(422, "conformance image builder returned a mutable image reference")
        evidence = app.state.image_build_authority.issue(receipt)
        stored = app.state.artifacts.put_json(evidence)
        repository.add_artifact(
            stored.id, experiment_id=None, project_id=row.project_id,
            kind="model-conformance-image-receipt", media_type=stored.media_type,
            size_bytes=stored.size_bytes, path=str(stored.path), metadata={
                "model_id": model_id, "image": image, "base_image": request.base_image,
                "driver_package_artifact_id": artifact_id,
            },
        )
        app.state.trusted_model_conformance_images.add(image)
        return {"image": image, "receipt": receipt, "artifact_id": stored.id}

    @app.post("/api/gpu/preflight")
    def gpu_preflight(request: GPUPreflightRequest) -> dict[str, Any]:
        if request.plan.provider != "local-cuda":
            return {
                "available": False,
                "reasons": ["RunPod requires a live quote and typed launch confirmation"],
                "telemetry": {"provider": request.plan.provider},
            }
        from .gpu import LocalCUDAExecutionProvider
        result = LocalCUDAExecutionProvider().preflight(
            request.plan, model_path=Path(request.model_path),
        )
        return {"available": result.available, "reasons": result.reasons, "telemetry": result.telemetry}

    @app.get("/api/environments")
    def environments() -> list[dict[str, Any]]:
        return [_environment(row) for row in repository.list_environments()]

    @app.get("/api/environments/{environment_id:path}")
    def environment(environment_id: str) -> dict[str, Any]:
        row = repository.get_environment(environment_id)
        if row is None:
            raise HTTPException(404, "environment not found")
        return _environment(row, details=True)

    @app.post("/api/environments/{environment_id:path}/validate-harbor", status_code=201)
    def validate_harbor_environment(
        environment_id: str, request: HarborValidationRequest,
    ) -> dict[str, Any]:
        row = repository.get_environment(environment_id)
        if row is None:
            raise HTTPException(404, "environment not found")
        try:
            package = EnvironmentPackageV1.model_validate_json(row.package_json)
        except ValueError as exc:
            raise HTTPException(422, f"invalid environment package: {exc}") from exc
        if package.environment_type != "coding" or not (
            package.id.startswith("harbor/") or package.id.startswith("repo2rlenv/")
        ):
            raise HTTPException(422, "environment is not an imported Harbor coding environment")
        source_uri = request.source_uri or package.source.uri
        try:
            source, _revision = app.state.environment_foundry.materializer.materialize(source_uri)
            root = source if source.is_dir() else source.parent
            from ..adapters.harbor import HarborEnvironmentAdapter
            from ..integrations.harbor_sandbox import DockerHarborTransport, HarborSandboxValidator
            rebuilt, bundles = HarborEnvironmentAdapter(app.state.artifacts).convert(
                root, environment_id=package.id, version=package.version,
                source_uri=package.source.uri, revision=package.source.revision,
            )
            if rebuilt.content_hash != package.content_hash:
                raise HTTPException(409, "Harbor source no longer matches the registered environment hash")
            report = HarborSandboxValidator(
                app.state.harbor_transport or DockerHarborTransport()
            ).validate(
                package, bundles, policy=request.policy,
            )
        except HTTPException:
            raise
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
            raise HTTPException(422, f"Harbor validation failed: {exc}") from exc
        repository.set_environment_validation(environment_id, report)
        stored = app.state.artifacts.put_json(report)
        repository.add_artifact(
            stored.id, experiment_id=None, project_id=row.project_id,
            kind="environment-validation", media_type=stored.media_type,
            size_bytes=stored.size_bytes, path=str(stored.path),
            metadata={"environment_id": row.id, "environment_hash": row.content_hash},
        )
        verifier_payload = app.state.artifacts.get_bytes(package.protected_verifier.artifact_id)
        verifier_hash = sha256(verifier_payload).hexdigest()
        evidence = app.state.promotion_authority.issue(
            environment_hash=row.content_hash,
            verifier_hash_before=verifier_hash,
            verifier_hash_after=verifier_hash,
            validation=report,
        )
        evidence_artifact = app.state.artifacts.put_json(evidence.model_dump(mode="json"))
        repository.add_artifact(
            evidence_artifact.id, experiment_id=None, project_id=row.project_id,
            kind="promotion-evidence", media_type=evidence_artifact.media_type,
            size_bytes=evidence_artifact.size_bytes, path=str(evidence_artifact.path),
            metadata={"environment_id": row.id, "environment_hash": row.content_hash,
                      "worker_id": evidence.worker_id},
        )
        promotion = app.state.promotion_authority.verify(evidence)
        promoted = None
        if promotion.promoted:
            final = promotion.model_copy(update={"evidence_artifact_id": evidence_artifact.id})
            promoted = repository.promote_environment(environment_id, final.model_dump(mode="json"))
        return {
            "artifact_id": stored.id,
            "evidence_artifact_id": evidence_artifact.id,
            "validation": report,
            "promotion": promotion.model_dump(mode="json"),
            "environment": _environment(promoted, details=True) if promoted else None,
        }

    @app.post("/api/environments/{environment_id:path}/sample")
    def sample_episode(environment_id: str) -> dict[str, Any]:
        row = repository.get_environment(environment_id)
        if row is None:
            raise HTTPException(404, "environment not found")
        if not environment_id.startswith("audioenv/reference-agentic"):
            raise HTTPException(422, "sample execution is not available for this environment adapter")
        from ..runtime import AudioEnvironment
        package = build_reference_package()
        task = next(task for task in package.tasks if task.split == "test")
        env = AudioEnvironment(package)
        env.reset(task.id, seed=30_000, model_id="sample-noop")
        env.step(package.public_action_candidates[task.id][0][0])
        return env.trace.to_dict(include_observability=False)

    @app.post("/api/environments/custom", status_code=201)
    def create_custom_environment(request: CustomTaskInput) -> dict[str, Any]:
        authored = DeterministicEnvironmentAuthor(app.state.artifacts).author(request)
        try:
            report = app.state.validation_runner.validate_custom(request)
        except IsolatedValidationError as exc:
            raise HTTPException(422, f"isolated environment validation failed: {exc}") from exc
        if report.get("environment_hash") != authored.boundary.content_hash:
            raise HTTPException(409, "isolated validator environment hash mismatch")
        if report.get("verifier_hash") != authored.boundary.protected_verifier.sha256:
            raise HTTPException(409, "isolated validator verifier hash mismatch")
        boundary = authored.boundary.model_dump(mode="json")
        row = repository.register_environment(
            record_id=f"{authored.boundary.id}@{authored.boundary.version}",
            version=authored.boundary.version,
            content_hash=authored.boundary.content_hash,
            source_kind=authored.boundary.source.kind,
            authoring_status=authored.boundary.authoring_status,
            package=boundary,
            validation=report,
        )
        return {"id": row.id, "package": boundary, "validation": report}

    @app.get("/api/artifacts")
    def artifacts() -> list[dict[str, Any]]:
        return [_artifact(row) for row in repository.list_artifacts()]

    @app.get("/api/artifacts/{artifact_id}/download")
    def download_artifact(artifact_id: str):
        row = repository.get_artifact(artifact_id)
        if row is None:
            raise HTTPException(404, "artifact not found")
        path = Path(row.path).resolve()
        if not path.is_file() or app.state.data_root not in path.parents:
            raise HTTPException(404, "artifact file unavailable")
        if sha256(path.read_bytes()).hexdigest() != row.id:
            raise HTTPException(409, "artifact failed content-hash verification")
        return FileResponse(path, media_type=row.media_type, filename=path.name)

    @app.get("/api/experiments/{experiment_id}/traces")
    def experiment_traces(experiment_id: str, limit: int = 50) -> dict[str, Any]:
        if repository.get_experiment(experiment_id) is None:
            raise HTTPException(404, "experiment not found")
        if limit < 1 or limit > 200:
            raise HTTPException(422, "trace limit must be between 1 and 200")
        candidates = [
            row for row in repository.list_artifacts()
            if row.experiment_id == experiment_id and row.kind == "training_trajectories"
        ]
        if not candidates:
            return {"experiment_id": experiment_id, "available": False, "traces": []}
        rows = []
        for artifact in candidates:
            path = Path(artifact.path).resolve()
            if not path.is_file() or app.state.data_root not in path.parents:
                raise HTTPException(404, "trajectory artifact file unavailable")
            payload = path.read_bytes()
            if sha256(payload).hexdigest() != artifact.id:
                raise HTTPException(409, "trajectory artifact failed content-hash verification")
            for line in payload.splitlines():
                try:
                    value = json.loads(line)
                    rows.append(_public_trace_row(value))
                except (ValueError, json.JSONDecodeError) as exc:
                    raise HTTPException(409, "trajectory artifact contains an invalid record") from exc
        rows = rows[-limit:]
        return {"experiment_id": experiment_id, "available": True,
                "artifact_id": candidates[0].id, "artifact_ids": [item.id for item in candidates],
                "traces": rows}

    @app.get("/api/evaluations")
    def evaluations() -> list[dict[str, Any]]:
        return [_evaluation(repository, row) for row in repository.list_evaluations()]

    @app.get("/api/evaluations/{evaluation_id}")
    def evaluation(evaluation_id: str) -> dict[str, Any]:
        row = repository.get_evaluation(evaluation_id)
        if row is None:
            raise HTTPException(404, "evaluation not found")
        return _evaluation(repository, row, samples=True)

    @app.get("/api/comparisons")
    def comparisons(experiment_id: str | None = None) -> list[dict[str, Any]]:
        return [{
            "id": row.id,
            "experiment_id": row.experiment_id,
            "baseline_evaluation_id": row.baseline_evaluation_id,
            "trained_evaluation_id": row.trained_evaluation_id,
            "compatibility_hash": row.compatibility_hash,
            "result": json.loads(row.result_json),
        } for row in repository.list_comparisons(experiment_id)]

    @app.post("/api/experiments/{experiment_id}/learning-evidence", status_code=201)
    def create_learning_evidence(
        experiment_id: str, request: LearningEvidenceRequest,
    ) -> dict[str, Any]:
        record = repository.get_experiment(experiment_id)
        if record is None:
            raise HTTPException(404, "experiment not found")
        identifiers = (
            request.baseline_evaluation_id, request.trained_evaluation_id,
            request.independent_baseline_evaluation_id,
            request.independent_trained_evaluation_id,
        )
        rows = [repository.get_evaluation(value) if value else None for value in identifiers]
        if rows[0] is None or rows[1] is None:
            raise HTTPException(422, "baseline and trained evaluations are required")
        if any(row is not None and row.experiment_id != experiment_id for row in rows):
            raise HTTPException(422, "all evaluations must belong to the experiment")
        if (rows[2] is None) != (rows[3] is None):
            raise HTTPException(422, "independent baseline and trained evaluations must be supplied together")
        baseline = _evaluation_result(repository, rows[0])
        trained = _evaluation_result(repository, rows[1])
        independent_baseline = _evaluation_result(repository, rows[2]) if rows[2] else None
        independent_trained = _evaluation_result(repository, rows[3]) if rows[3] else None
        reward_parity = _verified_reward_parity(
            app, repository, experiment_id, request.reward_parity_artifact_id,
            baseline.protocol.protocol_hash, trained.protocol.protocol_hash,
        )
        baseline_metrics = json.loads(rows[0].metrics_json)
        trained_metrics = json.loads(rows[1].metrics_json)
        evidence = assess_learning_evidence(
            baseline, trained,
            independent_baseline=independent_baseline,
            independent_trained=independent_trained,
            reward_parity_verified=reward_parity,
            baseline_speech_quality=baseline_metrics.get("speech_quality"),
            trained_speech_quality=trained_metrics.get("speech_quality"),
        )
        stored = app.state.artifacts.put_json(evidence.model_dump(mode="json"))
        artifact = repository.add_artifact(
            stored.id, experiment_id=experiment_id, project_id=record.project_id,
            kind="learning-evidence", media_type=stored.media_type,
            size_bytes=stored.size_bytes, path=str(stored.path),
            metadata={"status": evidence.status, "production_eligible": evidence.production_eligible},
        )
        repository.add_log(
            experiment_id, "Learning-evidence decision created",
            payload={"artifact_id": stored.id, "status": evidence.status},
        )
        return {"artifact": _artifact(artifact), "evidence": evidence.model_dump(mode="json")}

    @app.post("/api/experiments", status_code=201)
    def create_experiment(request: ExperimentCreate) -> dict[str, Any]:
        if request.project_id and repository.get_project(request.project_id) is None:
            raise HTTPException(422, "project not found")
        try:
            _assert_no_embedded_secrets(request.config)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return _public_experiment(repository.create_experiment(
            request.name, request.config, project_id=request.project_id,
        ))

    @app.post("/api/experiments/{experiment_id}/resume", status_code=201)
    def resume_experiment(experiment_id: str, request: ExperimentResume) -> dict[str, Any]:
        source = repository.get_experiment(experiment_id)
        if source is None:
            raise HTTPException(404, "experiment not found")
        if source.status not in {"completed", "failed", "cancelled"}:
            raise HTTPException(409, "only a terminal experiment can be resumed")
        artifact = repository.get_artifact(request.checkpoint_artifact_id)
        if artifact is None or artifact.experiment_id != experiment_id or artifact.kind != "checkpoint":
            raise HTTPException(404, "checkpoint not found")
        checkpoint = Path(artifact.path).resolve()
        if not checkpoint.is_file() or app.state.data_root not in checkpoint.parents:
            raise HTTPException(409, "checkpoint artifact is unavailable")
        if sha256(checkpoint.read_bytes()).hexdigest() != artifact.id:
            raise HTTPException(409, "checkpoint artifact failed content-hash verification")

        config = json.loads(source.config_json)
        try:
            _assert_no_embedded_secrets(config)
        except ValueError as exc:
            raise HTTPException(409, f"source experiment contains unsafe persisted secrets: {exc}") from exc
        provider_id = str(config.get("provider_id", "local-subprocess"))
        artifact_metadata = json.loads(artifact.metadata_json)
        if provider_id != "local-subprocess" and not (
            artifact_metadata.get("verification_source") == "gpu-worker-checkpoint-reload"
            and artifact_metadata.get("checkpoint_reload_verified") is True
        ):
            raise HTTPException(
                409, "GPU resume requires successful persisted checkpoint reload evidence",
            )
        hyperparameters = dict(config.get("hyperparameters") or config.get("trainer_config") or {})
        if provider_id == "local-subprocess":
            hyperparameters.update({"steps": request.additional_steps, "resume_from": str(checkpoint)})
        else:
            gpu = dict(config.get("gpu_worker") or {})
            gpu["resume_from"] = _remote_resume_checkpoint(
                repository, source.id, provider_id, artifact,
            )
            config["gpu_worker"] = gpu
            config["resume_storage"] = _resume_storage_binding(
                repository, source.id, provider_id,
            )
            hyperparameters["max_steps"] = request.additional_steps
            hyperparameters.pop("steps", None)
        config.update({
            "hyperparameters": hyperparameters,
            "resume_parent_experiment_id": source.id,
            "resume_checkpoint_artifact_id": artifact.id,
        })
        child = repository.create_experiment(request.run_name, config, project_id=source.project_id)
        lineage = {
            "parent_experiment_id": source.id,
            "checkpoint_artifact_id": artifact.id,
            "additional_steps": request.additional_steps,
        }
        repository.add_log(child.id, "Resume draft created from immutable checkpoint", payload=lineage)
        repository.add_log(source.id, "Child resume draft created", payload={
            **lineage, "child_experiment_id": child.id,
        })
        return _public_experiment(child)

    @app.get("/api/experiments")
    def list_experiments() -> list[dict[str, Any]]:
        return [_public_experiment(row) for row in repository.list_experiments()]

    @app.get("/api/experiments/{experiment_id}")
    def get_experiment(experiment_id: str) -> dict[str, Any]:
        record = repository.get_experiment(experiment_id)
        if record is None:
            raise HTTPException(404, "experiment not found")
        return _public_experiment(record)

    @app.post("/api/experiments/{experiment_id}/preflight")
    def preflight(experiment_id: str) -> dict[str, Any]:
        record = repository.get_experiment(experiment_id)
        if record is None:
            raise HTTPException(404, "experiment not found")
        result = _preflight(repository, json.loads(record.config_json))
        repository.set_preflight(experiment_id, result)
        return result

    @app.post("/api/experiments/{experiment_id}/transition")
    def transition(experiment_id: str, request: TransitionRequest) -> dict[str, Any]:
        try:
            return _public_experiment(repository.transition(experiment_id, request.target, request.payload))
        except KeyError:
            raise HTTPException(404, "experiment not found") from None
        except InvalidTransition as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/experiments/{experiment_id}/cancel")
    def cancel(experiment_id: str) -> dict[str, Any]:
        try:
            record = repository.request_cancel(experiment_id)
        except KeyError:
            raise HTTPException(404, "experiment not found") from None
        except InvalidTransition as exc:
            raise HTTPException(409, str(exc)) from exc
        active = [
            item for item in repository.list_compute_resources(experiment_id)
            if item.provider in {"runpod-secure-cloud", "modal-sandbox", "local-cuda"}
            and item.external_id and item.status in {"provisioning", "running", "cancelling"}
        ]
        if active:
            for resource in active:
                try:
                    if resource.provider == "runpod-secure-cloud":
                        from .runpod import RunPodConfig, RunPodRESTV2Transport, RunPodSecureCloudProvider
                        snapshot = json.loads(resource.config_json)
                        runpod_config = RunPodConfig(**snapshot["config"])
                        provider = RunPodSecureCloudProvider(
                            app.state.runpod_transport or RunPodRESTV2Transport()
                        )
                        remote = provider.reconnect(resource.external_id or "")
                        executor = _resolve_runpod_executor(app, remote, runpod_config, snapshot)
                        if executor is not None and "worker_launch" in snapshot:
                            provider.request_cancellation(executor, experiment_id)
                        provider.cancel(resource.external_id or "")
                    elif resource.provider == "modal-sandbox":
                        _modal_provider(app, json.loads(resource.config_json)).cancel(resource.external_id or "")
                    else:
                        provider = _local_cuda_provider(app)
                        provider.cancel(resource.external_id or "")
                        provider.cleanup(resource.external_id or "")
                    repository.update_compute_resource(resource.id, status="cancelled")
                except Exception as exc:
                    repository.update_compute_resource(resource.id, status="cancellation_failed")
                    repository.add_log(
                        experiment_id, "External compute cancellation failed", level="error",
                        payload={"resource_id": resource.id, "reason": str(exc)},
                    )
                    raise HTTPException(502, f"External compute cancellation failed: {exc}") from exc
            try:
                record = repository.transition(experiment_id, "cancelled", {
                    "provider": active[0].provider,
                    "resource_ids": [item.id for item in active],
                })
            except InvalidTransition as exc:
                raise HTTPException(409, str(exc)) from exc
        return _public_experiment(record)

    @app.get("/api/experiments/{experiment_id}/compute-resources")
    def compute_resources(experiment_id: str) -> list[dict[str, Any]]:
        if repository.get_experiment(experiment_id) is None:
            raise HTTPException(404, "experiment not found")
        return [_compute_resource(row) for row in repository.list_compute_resources(experiment_id)]

    @app.post("/api/experiments/{experiment_id}/compute-resources/{resource_id}/reconcile")
    def reconcile_compute_resource(experiment_id: str, resource_id: str) -> dict[str, Any]:
        resource = repository.get_compute_resource(resource_id)
        if resource is None or resource.experiment_id != experiment_id:
            raise HTTPException(404, "compute resource not found")
        if resource.provider == "modal-sandbox" and resource.external_id:
            snapshot = json.loads(resource.config_json)
            try:
                remote = _modal_provider(app, snapshot).reconnect(resource.external_id)
            except Exception as exc:
                raise HTTPException(502, f"Modal state refresh failed: {exc}") from exc
            state = str(remote.get("state", "unknown")).lower()
            status = "running" if state == "started" else (
                "stopped" if state == "finished" and remote.get("returncode") in {0, None}
                else "failed"
            )
            snapshot["last_remote_state"] = remote
            return _compute_resource(repository.update_compute_resource(
                resource.id, status=status, config=snapshot,
            ))
        if resource.provider == "local-cuda" and resource.external_id:
            snapshot = json.loads(resource.config_json)
            try:
                remote = _local_cuda_provider(app).reconnect(resource.external_id)
            except Exception as exc:
                raise HTTPException(502, f"Local CUDA state refresh failed: {exc}") from exc
            status = "running" if remote.get("running") else (
                "stopped" if remote.get("exit_code") == 0 else "failed"
            )
            snapshot["last_container_state"] = remote
            return _compute_resource(repository.update_compute_resource(
                resource.id, status=status, config=snapshot,
            ))
        if resource.provider != "runpod-secure-cloud" or not resource.external_id:
            raise HTTPException(422, "compute resource cannot be reconciled")
        from .runpod import RunPodRESTV2Transport, RunPodSecureCloudProvider
        provider = RunPodSecureCloudProvider(
            app.state.runpod_transport or RunPodRESTV2Transport()
        )
        try:
            remote = provider.reconnect(resource.external_id)
        except Exception as exc:
            raise HTTPException(502, f"RunPod state refresh failed: {exc}") from exc
        remote_status = str(remote.get("status", remote.get("desiredStatus", "unknown"))).lower()
        mapped = {
            "running": "provisioning", "provisioning": "provisioning",
            "starting": "provisioning", "stopped": "stopped", "exited": "stopped",
            "terminated": "stopped", "created": "provisioning", "pending": "provisioning",
            "error": "failed",
        }.get(remote_status, resource.status)
        snapshot = json.loads(resource.config_json)
        if remote_status == "running" and "worker_launch" in snapshot:
            mapped = "running"
        snapshot["last_remote_state"] = {
            key: remote[key] for key in (
                "id", "status", "desiredStatus", "publicIp", "ssh_host", "sshHost",
                "ssh_port", "sshPort", "ssh",
            ) if key in remote
        }
        resource = repository.update_compute_resource(resource.id, status=mapped, config=snapshot)
        if remote_status == "running" and "worker_launch" not in snapshot:
            try:
                resource = _start_runpod_worker(app, repository, resource, remote)
                repository.add_log(experiment_id, "RunPod GPU worker launched after reconnect", payload={
                    "resource_id": resource.id, "external_id": resource.external_id,
                })
            except Exception as exc:
                try:
                    provider.cleanup(resource.external_id)
                finally:
                    repository.update_compute_resource(resource.id, status="failed")
                raise HTTPException(502, f"RunPod worker launch failed; Pod stopped: {exc}") from exc
        return _compute_resource(resource)

    @app.post("/api/experiments/{experiment_id}/compute-resources/{resource_id}/sync")
    def sync_compute_resource(experiment_id: str, resource_id: str) -> dict[str, Any]:
        record = repository.get_experiment(experiment_id)
        if record is None:
            raise HTTPException(404, "experiment not found")
        resource = repository.get_compute_resource(resource_id)
        if resource is None or resource.experiment_id != experiment_id:
            raise HTTPException(404, "compute resource not found")
        if resource.provider == "modal-sandbox" and resource.external_id:
            return _sync_modal_resource(app, repository, record, resource)
        if resource.provider == "local-cuda" and resource.external_id:
            return _sync_local_cuda_resource(app, repository, record, resource)
        if resource.provider != "runpod-secure-cloud" or not resource.external_id:
            raise HTTPException(422, "compute resource cannot be synchronized")
        snapshot = json.loads(resource.config_json)
        if "worker_launch" not in snapshot:
            raise HTTPException(409, "GPU worker has not launched")
        from .runpod import RunPodConfig, RunPodQuote, RunPodRESTV2Transport, RunPodSecureCloudProvider
        runpod_config = RunPodConfig(**snapshot["config"])
        reviewed_quote = (
            RunPodQuote(**snapshot["quote"])
            if "quote" in snapshot else None
        )
        provider = RunPodSecureCloudProvider(
            app.state.runpod_transport
            or RunPodRESTV2Transport(api_key_env=runpod_config.api_key_secret_name)
        )
        try:
            pod = provider.reconnect(resource.external_id)
            executor = _resolve_runpod_executor(app, pod, runpod_config, snapshot)
            elapsed_hours = _elapsed_provider_hours(snapshot, resource.created_at)
            stopped = (
                provider.enforce_limits(
                    resource.external_id, runpod_config, reviewed_quote,
                    elapsed_hours=elapsed_hours,
                )
                if reviewed_quote is not None else None
            )
            if stopped is not None:
                if executor is not None:
                    try:
                        provider.request_cancellation(executor, experiment_id)
                    except Exception:
                        pass
                reason = "RunPod maximum runtime or budget reached"
                snapshot["limit_stop"] = {
                    "reason": reason, "elapsed_hours": elapsed_hours,
                    "stopped_at": datetime.now(UTC).isoformat(),
                }
                _advance_gpu_terminal(repository, experiment_id, "cancelled", {"reason": reason})
                resource = repository.update_compute_resource(
                    resource.id, status="cancelled", config=snapshot,
                )
                repository.add_log(experiment_id, reason, level="warning", payload={
                    "resource_id": resource.id, "elapsed_hours": elapsed_hours,
                })
                return {
                    "resource": _compute_resource(resource), "events_synced": 0,
                    "artifact_ids": [],
                }
            if executor is None:
                raise RuntimeError("SSH endpoint or key/known-host references are unavailable")
            after = int(snapshot.get("synced_event_sequence", 0))
            events = provider.stream_events(
                executor, experiment_id=experiment_id, after_sequence=after,
            )
            terminal = None
            for event in events:
                if event.type == "metric":
                    repository.add_metric(
                        experiment_id, int(event.payload["step"]), str(event.payload["name"]),
                        float(event.payload["value"]), str(event.payload.get("source", "gpu-worker")),
                    )
                elif event.type == "log":
                    repository.add_log(
                        experiment_id, str(event.payload.get("message", "")),
                        payload={"gpu_event_sequence": event.sequence,
                                 "channel": event.payload.get("channel")},
                    )
                else:
                    repository.add_log(
                        experiment_id, event.type, payload={
                            "gpu_event_sequence": event.sequence, **event.payload,
                        },
                    )
                if event.type in {"worker.completed", "worker.failed", "worker.cancelled"}:
                    terminal = event
            if events:
                snapshot["synced_event_sequence"] = events[-1].sequence
            artifacts: list[str] = []
            if terminal and terminal.type == "worker.completed":
                incoming = app.state.data_root / "incoming" / resource.id
                for path in provider.collect_artifacts(
                    executor, experiment_id=experiment_id, destination=incoming,
                ):
                    artifacts.append(_store_gpu_artifact(app, repository, record, path, incoming))
                _advance_gpu_terminal(repository, experiment_id, "completed")
                provider.cleanup(resource.external_id)
                resource = repository.update_compute_resource(resource.id, status="stopped", config=snapshot)
            elif terminal and terminal.type == "worker.failed":
                _advance_gpu_terminal(repository, experiment_id, "failed", terminal.payload)
                provider.cleanup(resource.external_id)
                resource = repository.update_compute_resource(resource.id, status="failed", config=snapshot)
            elif terminal and terminal.type == "worker.cancelled":
                _advance_gpu_terminal(repository, experiment_id, "cancelled", terminal.payload)
                provider.cleanup(resource.external_id)
                resource = repository.update_compute_resource(resource.id, status="cancelled", config=snapshot)
            else:
                current = repository.get_experiment(experiment_id)
                if current and current.status == "queued" and events:
                    repository.transition(experiment_id, "running", {"resource_id": resource.id})
                resource = repository.update_compute_resource(resource.id, status="running", config=snapshot)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(502, f"RunPod synchronization failed: {exc}") from exc
        return {
            "resource": _compute_resource(resource), "events_synced": len(events),
            "artifact_ids": artifacts,
        }

    @app.post("/api/experiments/{experiment_id}/runpod/quote", status_code=201)
    def runpod_quote(experiment_id: str, request: RunPodQuoteRequest) -> dict[str, Any]:
        record = repository.get_experiment(experiment_id)
        if record is None:
            raise HTTPException(404, "experiment not found")
        config = json.loads(record.config_json)
        reasons = _runpod_configuration_reasons(repository, config)
        _enforce_resume_storage(config, "runpod-secure-cloud", request.network_volume_id)
        if reasons:
            raise HTTPException(422, {"reasons": reasons})
        from .runpod import RunPodConfig, RunPodRESTV2Transport, RunPodSecureCloudProvider
        provider = RunPodSecureCloudProvider(
            app.state.runpod_transport or RunPodRESTV2Transport(api_key_env=request.api_key_secret_name)
        )
        request_values = request.model_dump()
        known_hosts_secret_name = request_values.pop("known_hosts_secret_name")
        runpod_config = RunPodConfig(run_name=record.name, **request_values)
        try:
            worker_experiment = _gpu_worker_experiment(record.id, record.name, config)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        try:
            quote = provider.quote(runpod_config)
        except (OSError, RuntimeError, ValueError) as exc:
            raise HTTPException(422, f"RunPod quote failed: {exc}") from exc
        if not quote.available:
            raise HTTPException(422, "no Secure Cloud GPU configuration satisfies availability and budget")
        snapshot = {
            "config": asdict(runpod_config), "quote": asdict(quote),
            "experiment_config_hash": record.config_hash,
            "known_hosts_secret_name": known_hosts_secret_name,
            "worker_experiment": worker_experiment.model_dump(mode="json"),
        }
        resource = repository.create_compute_resource(
            experiment_id=experiment_id, provider=provider.provider_id,
            status="quoted", config=snapshot,
        )
        repository.add_log(experiment_id, "RunPod quote persisted for review", payload={
            "resource_id": resource.id, "hourly_price_usd": quote.hourly_price_usd,
            "maximum_cost_usd": quote.maximum_cost_usd,
        })
        return _compute_resource(resource)

    @app.post("/api/experiments/{experiment_id}/local-cuda/prepare", status_code=201)
    def local_cuda_prepare(experiment_id: str, request: LocalCUDAPrepareRequest) -> dict[str, Any]:
        record = repository.get_experiment(experiment_id)
        if record is None:
            raise HTTPException(404, "experiment not found")
        config = json.loads(record.config_json)
        reasons = _gpu_configuration_reasons(repository, config, "local-cuda")
        _enforce_resume_storage(config, "local-cuda", str(Path(request.data_dir).resolve()))
        if request.plan.provider != "local-cuda":
            reasons.append("execution plan must target local-cuda")
        if reasons:
            raise HTTPException(422, {"reasons": reasons})
        provider = _local_cuda_provider(app)
        data_dir = Path(request.data_dir).resolve()
        model_path = Path(request.model_path).resolve()
        try:
            preflight = provider.preflight(request.plan, model_path=model_path)
            worker_experiment = _gpu_worker_experiment(record.id, record.name, config)
            if preflight.available:
                staged_path = provider.stage(
                    request.plan, data_dir=data_dir, experiment=worker_experiment,
                )
            else:
                staged_path = None
        except (OSError, RuntimeError, ValueError) as exc:
            raise HTTPException(422, f"Local CUDA preparation failed: {exc}") from exc
        snapshot = {
            "plan": request.plan.model_dump(mode="json"), "data_dir": str(data_dir),
            "model_path": str(model_path), "experiment_config_hash": record.config_hash,
            "worker_experiment": worker_experiment.model_dump(mode="json"),
            "preflight": {"available": preflight.available, "reasons": list(preflight.reasons),
                          "telemetry": preflight.telemetry},
            "experiment_path": str(staged_path) if staged_path else None,
        }
        resource = repository.create_compute_resource(
            experiment_id=experiment_id, provider=provider.provider_id,
            status="prepared" if preflight.available else "blocked", config=snapshot,
        )
        repository.add_log(experiment_id, "Local CUDA preflight persisted", payload={
            "resource_id": resource.id, **snapshot["preflight"],
        })
        return _compute_resource(resource)

    @app.post("/api/experiments/{experiment_id}/modal/quote", status_code=201)
    def modal_quote(experiment_id: str, request: ModalQuoteRequest) -> dict[str, Any]:
        record = repository.get_experiment(experiment_id)
        if record is None:
            raise HTTPException(404, "experiment not found")
        config = json.loads(record.config_json)
        reasons = _gpu_configuration_reasons(repository, config, "modal-sandbox")
        _enforce_resume_storage(config, "modal-sandbox", request.volume_name)
        if reasons:
            raise HTTPException(422, {"reasons": reasons})
        from .modal import ModalConfig, ModalSDKTransport, ModalSandboxProvider
        values = request.model_dump()
        hourly_rate = values.pop("hourly_price_per_gpu_usd")
        modal_config = ModalConfig(run_name=record.name, **values)
        transport = app.state.modal_transport or ModalSDKTransport(
            hourly_prices={modal_config.gpu_type: hourly_rate},
        )
        provider = ModalSandboxProvider(transport)
        try:
            worker_experiment = _gpu_worker_experiment(record.id, record.name, config)
            quote = provider.quote(modal_config)
        except (OSError, RuntimeError, ValueError) as exc:
            raise HTTPException(422, f"Modal quote failed: {exc}") from exc
        if not quote.available:
            raise HTTPException(422, "Modal GPU quote is unavailable or exceeds the configured budget")
        snapshot = {
            "config": asdict(modal_config), "quote": asdict(quote),
            "experiment_config_hash": record.config_hash,
            "worker_experiment": worker_experiment.model_dump(mode="json"),
            "pricing_basis": "operator-configured rate; Modal does not expose a live pricing quote API",
        }
        resource = repository.create_compute_resource(
            experiment_id=experiment_id, provider=provider.provider_id,
            status="quoted", config=snapshot,
        )
        repository.add_log(experiment_id, "Modal quote persisted for review", payload={
            "resource_id": resource.id, "hourly_price_usd": quote.hourly_price_usd,
            "maximum_cost_usd": quote.maximum_cost_usd,
            "pricing_basis": snapshot["pricing_basis"],
        })
        return _compute_resource(resource)

    @app.post("/api/experiments/{experiment_id}/launch", status_code=202)
    def launch(experiment_id: str, request: LaunchRequest) -> dict[str, Any]:
        if not request.confirmed:
            raise HTTPException(400, "launch requires explicit confirmation")
        record = repository.get_experiment(experiment_id)
        if record is None:
            raise HTTPException(404, "experiment not found")
        if request.provider == "runpod-secure-cloud":
            return _launch_runpod(
                app, repository, record, request.quote_id, request.confirmation_name,
            )
        if request.provider == "modal-sandbox":
            return _launch_modal(
                app, repository, record, request.quote_id, request.confirmation_name,
            )
        if request.provider == "local-cuda":
            return _launch_local_cuda(
                app, repository, record, request.compute_resource_id,
                request.confirmation_name,
            )
        if request.provider != "local-subprocess":
            raise HTTPException(422, "unsupported execution provider")
        config = json.loads(record.config_json)
        check = _preflight(repository, config)
        repository.set_preflight(experiment_id, check)
        if not check["available"]:
            raise HTTPException(422, {"preflight": check})
        if not resolved_url.startswith("sqlite:///"):
            raise HTTPException(422, "local subprocess launch currently requires SQLite")
        database_path = Path(resolved_url.removeprefix("sqlite:///"))
        from .execution import LocalExecutionProvider
        try:
            repository.transition(experiment_id, "validating")
            repository.transition(experiment_id, "queued")
        except InvalidTransition as exc:
            raise HTTPException(409, str(exc)) from exc
        repository.add_log(experiment_id, "Preflight passed; local worker queued", payload=check)
        job = repository.create_job(
            experiment_id,
            "small_audio_train",
            {
                "config": config.get("hyperparameters", config.get("trainer_config", {})),
                "output": str(data_root / "runs" / experiment_id),
            },
        )
        submitted = LocalExecutionProvider().submit(database=database_path, job_id=job.id)
        app.state.processes[job.id] = submitted.process
        return {"job_id": job.id, "pid": submitted.pid, "preflight": check, "cost": check["cost"]}

    @app.get("/api/experiments/{experiment_id}/logs")
    def logs(experiment_id: str) -> list[dict[str, Any]]:
        if repository.get_experiment(experiment_id) is None:
            raise HTTPException(404, "experiment not found")
        return [{
            "id": row.id, "level": row.level, "message": row.message,
            "payload": _redact_public(json.loads(row.payload_json)),
            "created_at": row.created_at.isoformat(),
        } for row in repository.list_logs(experiment_id)]

    @app.post("/api/experiments/{experiment_id}/export", status_code=201)
    def export_run(experiment_id: str) -> dict[str, Any]:
        record = repository.get_experiment(experiment_id)
        if record is None:
            raise HTTPException(404, "experiment not found")
        from .export import export_experiment
        path = export_experiment(repository, experiment_id, data_root / "exports" / experiment_id)
        payload = path.read_bytes()
        artifact = repository.add_artifact(
            sha256(payload).hexdigest(), experiment_id=experiment_id,
            project_id=record.project_id, kind="experiment_export",
            media_type="application/json", size_bytes=len(payload), path=str(path.resolve()),
            metadata={"config_hash": record.config_hash},
        )
        repository.add_log(experiment_id, "Reproducibility export created", payload={"artifact_id": artifact.id})
        return _artifact(artifact)

    @app.post("/api/experiments/{experiment_id}/checkpoints/{artifact_id}/verify")
    def verify_checkpoint(experiment_id: str, artifact_id: str) -> dict[str, Any]:
        row = repository.get_artifact(artifact_id)
        if row is None or row.experiment_id != experiment_id or row.kind != "checkpoint":
            raise HTTPException(404, "checkpoint not found")
        checkpoint = Path(row.path).resolve()
        if not checkpoint.is_file() or app.state.data_root not in checkpoint.parents:
            return {"verified": False, "artifact_id": artifact_id,
                    "reason": "checkpoint artifact is unavailable"}
        if sha256(checkpoint.read_bytes()).hexdigest() != row.id:
            return {"verified": False, "artifact_id": artifact_id,
                    "reason": "checkpoint artifact failed content-hash verification"}
        metadata = json.loads(row.metadata_json)
        if (metadata.get("verification_source") == "gpu-worker-checkpoint-reload"
                and metadata.get("checkpoint_reload_verified") is True):
            result = {
                "verified": True,
                "artifact_id": artifact_id,
                "source": metadata["verification_source"],
                "checkpoint_directory": metadata.get("checkpoint_directory"),
                "checkpoint_tree_sha256": metadata.get("checkpoint_tree_sha256"),
            }
            if metadata.get("step") is not None:
                result["step"] = int(metadata["step"])
            return result
        try:
            model, payload = TinyAudioPolicyTrainer(build_reference_package()).load_checkpoint(checkpoint)
            del model
            return {"verified": True, "artifact_id": artifact_id, "step": int(payload["step"]),
                    "source": "tiny-audio-policy-checkpoint-reload"}
        except Exception as exc:
            return {"verified": False, "artifact_id": artifact_id, "reason": str(exc)}

    @app.post("/api/experiments/{experiment_id}/metrics", status_code=201)
    def add_metric(experiment_id: str, request: MetricCreate) -> dict[str, Any]:
        try:
            row = repository.add_metric(experiment_id, request.step, request.name, request.value, request.source)
        except KeyError:
            raise HTTPException(404, "experiment not found") from None
        return {"id": row.id, "step": row.step, "name": row.name, "value": row.value, "source": row.source}

    @app.get("/api/experiments/{experiment_id}/metrics")
    def metrics(experiment_id: str) -> list[dict[str, Any]]:
        if repository.get_experiment(experiment_id) is None:
            raise HTTPException(404, "experiment not found")
        return [{
            "id": row.id, "step": row.step, "name": row.name,
            "value": row.value, "source": row.source,
        } for row in repository.list_metrics(experiment_id)]

    @app.get("/api/experiments/{experiment_id}/events")
    async def events(experiment_id: str) -> StreamingResponse:
        if repository.get_experiment(experiment_id) is None:
            raise HTTPException(404, "experiment not found")

        async def stream():
            last = None
            while True:
                record = repository.get_experiment(experiment_id)
                if record is None:
                    return
                encoded = json.dumps(_public_experiment(record), separators=(",", ":"))
                if encoded != last:
                    yield f"event: experiment\ndata: {encoded}\n\n"
                    last = encoded
                if record.status in {"completed", "failed", "cancelled"}:
                    return
                await asyncio.sleep(1)

        return StreamingResponse(stream(), media_type="text/event-stream")

    async def supervise_external_resources() -> None:
        interval = app.state.external_resource_poll_interval_seconds
        if interval is None or interval <= 0:
            return
        while True:
            await asyncio.sleep(interval)
            resources = repository.list_active_compute_resources()

            async def supervise(resource) -> None:
                try:
                    if resource.provider == "runpod-secure-cloud" and resource.status == "provisioning":
                        await asyncio.to_thread(
                            reconcile_compute_resource, resource.experiment_id, resource.id,
                        )
                        resource = repository.get_compute_resource(resource.id)
                        if resource is None or resource.status != "running":
                            return
                    await asyncio.to_thread(
                        sync_compute_resource, resource.experiment_id, resource.id,
                    )
                    app.state.external_resource_watchdog_errors.pop(resource.id, None)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    # A transient provider error must not disable supervision of other jobs.
                    app.state.external_resource_watchdog_errors[resource.id] = str(exc)

            await asyncio.gather(*(supervise(resource) for resource in resources))

    async def start_external_resource_watchdog() -> None:
        interval = app.state.external_resource_poll_interval_seconds
        if interval is not None and interval > 0:
            app.state.external_resource_watchdog_task = asyncio.create_task(
                supervise_external_resources(), name="audioenv-external-resource-watchdog",
            )

    async def stop_external_resource_watchdog() -> None:
        task = app.state.external_resource_watchdog_task
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
            app.state.external_resource_watchdog_task = None

    app.router.add_event_handler("startup", start_external_resource_watchdog)
    app.router.add_event_handler("shutdown", stop_external_resource_watchdog)

    if static_dir is not None and static_dir.is_dir():
        assets = static_dir / "assets"
        if assets.is_dir():
            app.mount("/assets", StaticFiles(directory=assets), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str):
            candidate = (static_dir / path).resolve()
            if path and static_dir.resolve() in candidate.parents and candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(static_dir / "index.html")

    return app


def _seed_builtin_records(repository: PlatformRepository) -> None:
    package = build_reference_package()
    environment_hash, verifier_hash = package_hashes(package)
    manifest = build_reference_split_manifest(package)
    env_id = f"{package.spec.id}@{package.spec.version}"
    if repository.get_environment(env_id) is None:
        report = validate_package(package)
        repository.register_environment(
            record_id=env_id,
            version=package.spec.version,
            content_hash=environment_hash,
            source_kind="builtin",
            authoring_status="validated",
            package={
                "id": package.spec.id,
                "version": package.spec.version,
                "description": package.spec.description,
                "task_count": len(package.tasks),
                "split_manifest_hash": manifest.content_hash,
                "verifier_hash": verifier_hash,
            },
            validation={"passed": report.passed, "checks": [asdict(check) for check in report.checks]},
        )
        repository.save_split_manifest(manifest.content_hash, env_id, manifest.model_dump(mode="json"))
    model_id = "audioenv/small-audio-tool-policy"
    builtin_capabilities = {
        "audio_input": True, "audio_output": False, "structured_tools": True,
        "token_logprobs": True, "trainable": True, "batching": False,
        "distributed": False, "trainable_modules": ["audio_candidate_scorer"],
        "checkpoint_format": "pytorch-state-dict",
    }
    if repository.get_model(model_id) is None:
        repository.register_model(
            model_id, "Small Audio Tool Policy", "small-audio-tool-policy", "agent",
            {"claim_scope": "audio_conditioned_platform_proof"}, registration_state="certified",
        )
        estimate = TinyAudioPolicyTrainer(package).preflight(TinyTrainerConfig(steps=1))
        repository.set_model_probe(model_id, {
            "available": estimate.available, "device": "cpu", "revision": "builtin-v2",
            "reasons": list(estimate.reasons), "capabilities": builtin_capabilities,
            "registration_state": "certified",
        })
    else:
        existing = repository.get_model(model_id)
        probe = json.loads(existing.probe_json) if existing and existing.probe_json else {}
        if "capabilities" not in probe or existing.registration_state != "certified":
            probe.update({
                "capabilities": builtin_capabilities,
                "registration_state": "certified",
                "available": bool(probe.get("available", True)),
            })
            repository.set_model_probe(model_id, probe)


def _preflight(repository: PlatformRepository, config: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    trainer_id = config.get("trainer_id", "audioenv/small-audio-reinforce")
    provider_id = config.get("provider_id", "local-subprocess")
    environment_id = config.get("environment_id")
    agent_id = config.get("agent_model_id", "audioenv/small-audio-tool-policy")
    if provider_id in {"runpod-secure-cloud", "modal-sandbox", "local-cuda"}:
        reasons = _gpu_configuration_reasons(repository, config, provider_id)
        label = {
            "runpod-secure-cloud": "Secure Cloud GPU selected by live quote",
            "modal-sandbox": "Modal GPU selected by reviewed estimate",
            "local-cuda": "Local NVIDIA GPU selected by machine preflight",
        }[provider_id]
        basis = {
            "runpod-secure-cloud": "live RunPod quote required before launch",
            "modal-sandbox": "operator-configured Modal rate required before launch",
            "local-cuda": "local hardware; machine preflight required before launch",
        }[provider_id]
        return {
            "available": not reasons, "reasons": reasons, "trainer_id": trainer_id,
            "provider_id": provider_id, "environment_id": environment_id,
            "agent_model_id": agent_id,
            "hardware": {"device_type": label, "device_count": 8},
            "cost": {"status": "quote_required", "estimated_usd": None,
                     "basis": basis},
            "resource_estimate": None, "launch_requires_quote": provider_id != "local-cuda",
            "gpu_acceptance_status": "not_run",
            "trainer_release_tier": (
                "experimental" if trainer_id == "qwen3-omni-thinker-grpo-lora"
                else "release_candidate"
            ),
        }
    if trainer_id not in {"audioenv/small-audio-reinforce", "audioenv/tiny-reinforce"}:
        reasons.append("trainer is not available on the local provider")
    if provider_id != "local-subprocess":
        reasons.append("RunPod GPU acceptance has not run")
    environment = repository.get_environment(environment_id) if environment_id else None
    if environment is None:
        reasons.append("select a registered environment version")
    elif environment.authoring_status != "validated":
        reasons.append("environment is not validated")
    model = repository.get_model(agent_id)
    if model is None or not model.available:
        reasons.append("agent model has not passed its capability probe")
    try:
        trainer_config = TinyTrainerConfig(**config.get("hyperparameters", config.get("trainer_config", {})))
        estimate = TinyAudioPolicyTrainer(build_reference_package()).preflight(trainer_config)
        reasons.extend(estimate.reasons)
    except Exception as exc:
        estimate = None
        reasons.append(str(exc))
    return {
        "available": not reasons,
        "reasons": reasons,
        "trainer_id": trainer_id,
        "provider_id": provider_id,
        "environment_id": environment_id,
        "agent_model_id": agent_id,
        "hardware": {"device_type": "cpu", "device_count": 1},
        "cost": {"status": "estimated", "estimated_usd": 0.0, "basis": "local CPU; electricity excluded"},
        "resource_estimate": estimate.model_dump(mode="json") if estimate else None,
    }


def _runpod_configuration_reasons(
    repository: PlatformRepository, config: dict[str, Any],
) -> list[str]:
    return _gpu_configuration_reasons(repository, config, "runpod-secure-cloud")


def _gpu_configuration_reasons(
    repository: PlatformRepository, config: dict[str, Any], provider_id: str,
) -> list[str]:
    reasons: list[str] = []
    if config.get("provider_id") != provider_id:
        reasons.append(f"experiment provider must be {provider_id}")
    trainer_id = config.get("trainer_id")
    if trainer_id not in {
        "qwen3-omni-thinker-gspo-lora", "qwen3-omni-thinker-grpo-lora",
    }:
        reasons.append("GPU Qwen execution requires the GSPO or GRPO LoRA trainer")
    environment = repository.get_environment(config.get("environment_id", ""))
    if environment is None or environment.authoring_status != "validated":
        reasons.append("select a validated environment version")
    model = repository.get_model(config.get("agent_model_id", ""))
    if model is None or model.adapter_type != "qwen3-omni":
        reasons.append("select a registered Qwen3-Omni agent model")
    try:
        _gpu_worker_experiment("preflight", "preflight", config)
    except ValueError as exc:
        reasons.append(str(exc))
    return reasons


def _gpu_worker_experiment(
    experiment_id: str, run_name: str, config: dict[str, Any],
) -> GPUWorkerExperimentV1:
    gpu = config.get("gpu_worker")
    if not isinstance(gpu, dict):
        raise ValueError("complete gpu_worker paths are required before a RunPod quote")
    trainer = config.get("trainer_id")
    algorithm = "gspo" if trainer == "qwen3-omni-thinker-gspo-lora" else "grpo"
    try:
        return GPUWorkerExperimentV1(
            experiment_id=experiment_id, run_name=run_name, backend="qwen3-omni-verl",
            trainer=trainer, algorithm=algorithm,
            environment_manifest=gpu.get("environment_manifest"),
            model_path=gpu.get("model_path"), caller_model_path=gpu.get("caller_model_path"),
            rollout_dataset_path=gpu.get("rollout_dataset_path"),
            validation_dataset_path=gpu.get("validation_dataset_path"),
            verl_checkout=gpu.get("verl_checkout"), tau2_checkout=gpu.get("tau2_checkout"),
            output_dir=f"/workspace/runs/{experiment_id}/output",
            cancellation_file=f"/workspace/runs/{experiment_id}/cancel.requested",
            hyperparameters=config.get("hyperparameters", {}),
            dependency_pins=gpu.get("dependency_pins", {}),
            resume_from=gpu.get("resume_from"),
        )
    except Exception as exc:
        raise ValueError(f"invalid gpu_worker experiment: {exc}") from exc


def _start_runpod_worker(app, repository: PlatformRepository, resource, pod: dict[str, Any]):
    from .runpod import RunPodConfig, RunPodSecureCloudProvider
    snapshot = json.loads(resource.config_json)
    runpod_config = RunPodConfig(**snapshot["config"])
    executor = _resolve_runpod_executor(app, pod, runpod_config, snapshot)
    if executor is None:
        return resource
    experiment = GPUWorkerExperimentV1.model_validate(snapshot["worker_experiment"])
    provider = RunPodSecureCloudProvider(app.state.runpod_transport)
    staged_path = provider.stage(executor, experiment)
    launch = provider.launch(executor, staged_path)
    snapshot["worker_launch"] = {"experiment_path": staged_path, **launch}
    resource = repository.update_compute_resource(resource.id, status="running", config=snapshot)
    return resource


def _resolve_runpod_executor(app, pod: dict[str, Any], runpod_config, snapshot: dict[str, Any]):
    from .runpod import OpenSSHRemoteExecutor
    if app.state.runpod_executor_factory is not None:
        return app.state.runpod_executor_factory(pod, runpod_config)
    ssh = pod.get("ssh")
    connection = None
    if isinstance(ssh, dict):
        for kind in ("direct", "proxy"):
            candidate = ssh.get(kind)
            if isinstance(candidate, dict) and candidate.get("host"):
                connection = candidate
                break
    host = (
        connection.get("host") if connection else
        pod.get("ssh_host", pod.get("sshHost", pod.get("publicIp")))
    )
    port = int(
        connection.get("port", 22) if connection else
        pod.get("ssh_port", pod.get("sshPort", 22))
    )
    user = str(connection.get("username", "root")) if connection else "root"
    key_value = os.environ.get(runpod_config.ssh_key_secret_name)
    known_value = os.environ.get(snapshot.get("known_hosts_secret_name", "RUNPOD_KNOWN_HOSTS"))
    if not host or not key_value or not known_value:
        return None
    return OpenSSHRemoteExecutor(
        host=str(host), port=port, user=user, key_path=Path(key_value),
        known_hosts_path=Path(known_value),
    )


def _advance_gpu_terminal(
    repository: PlatformRepository, experiment_id: str, target: str,
    payload: dict[str, Any] | None = None,
) -> None:
    current = repository.get_experiment(experiment_id)
    if current is None:
        raise KeyError(experiment_id)
    if current.status == "queued":
        current = repository.transition(experiment_id, "running", {"source": "gpu-worker"})
    if target == "completed":
        if current.status == "running":
            current = repository.transition(experiment_id, "evaluating", {"source": "gpu-worker"})
        if current.status == "evaluating":
            repository.transition(experiment_id, "completed", payload or {})
    elif target in {"failed", "cancelled"} and current.status not in {"completed", "failed", "cancelled"}:
        repository.transition(experiment_id, target, payload or {})


def _launch_runpod(app, repository: PlatformRepository, record, quote_id, confirmation_name):
    if confirmation_name != record.name:
        raise HTTPException(400, "typed confirmation must exactly match the run name")
    if not quote_id:
        raise HTTPException(422, "a persisted RunPod quote is required")
    resource = repository.get_compute_resource(quote_id)
    if (
        resource is None or resource.experiment_id != record.id
        or resource.provider != "runpod-secure-cloud" or resource.status != "quoted"
    ):
        raise HTTPException(422, "RunPod quote is missing, stale, or belongs to another experiment")
    snapshot = json.loads(resource.config_json)
    if snapshot.get("experiment_config_hash") != record.config_hash:
        raise HTTPException(409, "experiment configuration changed after the quote")
    from .runpod import (
        RunPodConfig, RunPodQuote, RunPodRESTV2Transport, RunPodSecureCloudProvider,
    )
    runpod_config = RunPodConfig(**snapshot["config"])
    reviewed_quote = RunPodQuote(**snapshot["quote"])
    provider = RunPodSecureCloudProvider(
        app.state.runpod_transport
        or RunPodRESTV2Transport(api_key_env=runpod_config.api_key_secret_name)
    )
    try:
        fresh_quote = provider.quote(runpod_config)
    except (OSError, RuntimeError, ValueError) as exc:
        raise HTTPException(422, f"RunPod quote refresh failed: {exc}") from exc
    if fresh_quote != reviewed_quote:
        updated = dict(snapshot)
        updated["refreshed_quote"] = asdict(fresh_quote)
        repository.update_compute_resource(resource.id, status="requote_required", config=updated)
        raise HTTPException(409, "RunPod price or availability changed; review a fresh quote")
    try:
        repository.transition(record.id, "validating")
        pod = provider.provision(
            runpod_config, fresh_quote, confirmation_name=confirmation_name or "",
        )
        external_id = str(pod.get("id", ""))
        if not external_id:
            raise RuntimeError("RunPod create response did not include a Pod id")
        snapshot["provisioned_at"] = datetime.now(UTC).isoformat()
        resource = repository.update_compute_resource(
            resource.id, status="provisioning", external_id=external_id, config=snapshot,
        )
        remote_status = str(pod.get("status", pod.get("desiredStatus", ""))).lower()
        if remote_status == "running":
            resource = _start_runpod_worker(app, repository, resource, pod)
        repository.transition(record.id, "queued", {
            "provider": provider.provider_id, "resource_id": resource.id,
            "external_id": external_id,
        })
    except InvalidTransition as exc:
        if resource.external_id:
            try:
                provider.cleanup(resource.external_id)
            except Exception:
                pass
            repository.update_compute_resource(resource.id, status="failed")
        raise HTTPException(409, str(exc)) from exc
    except Exception as exc:
        if resource.external_id:
            try:
                provider.cleanup(resource.external_id)
            except Exception:
                pass
        try:
            repository.transition(record.id, "failed", {"reason": f"RunPod provisioning failed: {exc}"})
        except InvalidTransition:
            pass
        repository.update_compute_resource(resource.id, status="failed")
        raise HTTPException(502, f"RunPod provisioning failed: {exc}") from exc
    message = (
        "RunPod GPU worker launched" if resource.status == "running"
        else "RunPod Pod provisioned; awaiting worker connection"
    )
    repository.add_log(record.id, message, payload={
        "resource_id": resource.id, "external_id": resource.external_id,
        "maximum_cost_usd": reviewed_quote.maximum_cost_usd,
    })
    return {
        "resource": _compute_resource(resource), "preflight": {
            "available": True, "provider_id": provider.provider_id,
            "hardware": {"device_type": reviewed_quote.gpu_type_id,
                         "device_count": reviewed_quote.gpu_count},
            "cost": {"status": "quoted", "estimated_usd": reviewed_quote.maximum_cost_usd,
                     "basis": f"{reviewed_quote.hourly_price_usd}/hour maximum"},
        },
    }


def _modal_provider(app, snapshot: dict[str, Any]):
    from .modal import ModalSDKTransport, ModalSandboxProvider
    if app.state.modal_transport is not None:
        return ModalSandboxProvider(app.state.modal_transport)
    config = snapshot.get("config", {})
    quote = snapshot.get("quote", {})
    count = max(int(config.get("gpu_count", 1)), 1)
    hourly = float(quote.get("hourly_price_usd", 0.0)) / count
    return ModalSandboxProvider(ModalSDKTransport(
        hourly_prices={str(config.get("gpu_type", "H100")): hourly},
    ))


def _local_cuda_provider(app):
    if app.state.local_cuda_provider is not None:
        return app.state.local_cuda_provider
    from .gpu import LocalCUDAExecutionProvider
    return LocalCUDAExecutionProvider()


def _launch_local_cuda(app, repository: PlatformRepository, record, resource_id, confirmation_name):
    if confirmation_name != record.name:
        raise HTTPException(400, "typed confirmation must exactly match the run name")
    if not resource_id:
        raise HTTPException(422, "a persisted local CUDA preflight is required")
    resource = repository.get_compute_resource(resource_id)
    if (
        resource is None or resource.experiment_id != record.id
        or resource.provider != "local-cuda" or resource.status != "prepared"
    ):
        raise HTTPException(422, "local CUDA preflight is missing, blocked, stale, or belongs to another experiment")
    snapshot = json.loads(resource.config_json)
    if snapshot.get("experiment_config_hash") != record.config_hash:
        raise HTTPException(409, "experiment configuration changed after local CUDA preflight")
    plan = GPUExecutionPlanV1.model_validate(snapshot["plan"])
    provider = _local_cuda_provider(app)
    container_id: str | None = None
    try:
        repository.transition(record.id, "validating")
        launch = provider.launch(
            plan, data_dir=Path(snapshot["data_dir"]),
            experiment_path=Path(snapshot["experiment_path"]), run_name=record.name,
            confirmation_name=confirmation_name or "",
        )
        container_id = str(launch["container_id"])
        snapshot["worker_launch"] = launch
        resource = repository.update_compute_resource(
            resource.id, status="running", external_id=container_id, config=snapshot,
        )
        repository.transition(record.id, "queued", {
            "provider": provider.provider_id, "resource_id": resource.id,
            "external_id": container_id,
        })
    except InvalidTransition as exc:
        if container_id:
            try:
                provider.cancel(container_id)
                provider.cleanup(container_id)
            except Exception:
                pass
        raise HTTPException(409, str(exc)) from exc
    except Exception as exc:
        if container_id:
            try:
                provider.cancel(container_id)
                provider.cleanup(container_id)
            except Exception:
                pass
        try:
            repository.transition(record.id, "failed", {"reason": f"Local CUDA launch failed: {exc}"})
        except InvalidTransition:
            pass
        repository.update_compute_resource(resource.id, status="failed")
        raise HTTPException(502, f"Local CUDA launch failed: {exc}") from exc
    repository.add_log(record.id, "Local CUDA GPU worker launched", payload={
        "resource_id": resource.id, "container_id": container_id,
    })
    return {
        "resource": _compute_resource(resource), "preflight": snapshot["preflight"],
        "cost": {"status": "unavailable", "estimated_usd": None,
                 "basis": "local hardware; electricity excluded"},
    }


def _sync_local_cuda_resource(app, repository: PlatformRepository, record, resource) -> dict[str, Any]:
    snapshot = json.loads(resource.config_json)
    provider = _local_cuda_provider(app)
    try:
        after = int(snapshot.get("synced_event_sequence", 0))
        events = provider.stream_events(
            resource.external_id, experiment_id=record.id, after_sequence=after,
        )
        terminal = _persist_gpu_events(repository, record.id, events)
        if events:
            snapshot["synced_event_sequence"] = events[-1].sequence
        artifacts: list[str] = []
        if terminal and terminal.type == "worker.completed":
            incoming = app.state.data_root / "incoming" / resource.id
            for path in provider.collect_artifacts(
                data_dir=Path(snapshot["data_dir"]), experiment_id=record.id,
                destination=incoming,
            ):
                artifacts.append(_store_gpu_artifact(app, repository, record, path, incoming))
            _advance_gpu_terminal(repository, record.id, "completed")
            provider.cancel(resource.external_id)
            provider.cleanup(resource.external_id)
            resource = repository.update_compute_resource(resource.id, status="stopped", config=snapshot)
        elif terminal and terminal.type == "worker.failed":
            _advance_gpu_terminal(repository, record.id, "failed", terminal.payload)
            provider.cancel(resource.external_id)
            provider.cleanup(resource.external_id)
            resource = repository.update_compute_resource(resource.id, status="failed", config=snapshot)
        elif terminal and terminal.type == "worker.cancelled":
            _advance_gpu_terminal(repository, record.id, "cancelled", terminal.payload)
            provider.cancel(resource.external_id)
            provider.cleanup(resource.external_id)
            resource = repository.update_compute_resource(resource.id, status="cancelled", config=snapshot)
        else:
            current = repository.get_experiment(record.id)
            if current and current.status == "queued" and events:
                repository.transition(record.id, "running", {"resource_id": resource.id})
            resource = repository.update_compute_resource(resource.id, status="running", config=snapshot)
    except Exception as exc:
        raise HTTPException(502, f"Local CUDA synchronization failed: {exc}") from exc
    return {"resource": _compute_resource(resource), "events_synced": len(events), "artifact_ids": artifacts}


def _persist_gpu_events(repository: PlatformRepository, experiment_id: str, events):
    terminal = None
    for event in events:
        if event.type == "metric":
            repository.add_metric(
                experiment_id, int(event.payload["step"]), str(event.payload["name"]),
                float(event.payload["value"]), str(event.payload.get("source", "gpu-worker")),
            )
        elif event.type == "log":
            repository.add_log(experiment_id, str(event.payload.get("message", "")), payload={
                "gpu_event_sequence": event.sequence, "channel": event.payload.get("channel"),
            })
        else:
            repository.add_log(experiment_id, event.type, payload={
                "gpu_event_sequence": event.sequence, **event.payload,
            })
        if event.type in {"worker.completed", "worker.failed", "worker.cancelled"}:
            terminal = event
    return terminal


def _store_gpu_artifact(app, repository: PlatformRepository, record, path: Path, incoming: Path) -> str:
    payload = path.read_bytes()
    media_type = "application/json" if path.suffix in {".json", ".jsonl"} else "application/octet-stream"
    stored = app.state.artifacts.put_bytes(payload, media_type=media_type)
    kind = _gpu_artifact_kind(path)
    relative = str(path.relative_to(incoming))
    metadata: dict[str, Any] = {"remote_relative_path": relative}
    if kind == "checkpoint":
        metadata.update(_gpu_checkpoint_evidence(repository, record.id, relative))
    row = repository.add_artifact(
        stored.id, experiment_id=record.id, project_id=record.project_id,
        kind=kind, media_type=media_type, size_bytes=stored.size_bytes,
        path=str(stored.path), metadata=metadata,
    )
    return row.id


def _gpu_artifact_kind(path: Path) -> str:
    if ("checkpoint" in path.parts or "adapter" in path.name
            or any(part.startswith("global_step_") for part in path.parts)):
        return "checkpoint"
    if "trajectories" in path.parts and path.name.startswith("trajectory-") and path.suffix == ".jsonl":
        return "training_trajectories"
    return "gpu_run_artifact"


def _gpu_checkpoint_evidence(
    repository: PlatformRepository, experiment_id: str, relative_path: str,
) -> dict[str, Any]:
    """Bind a collected file to the worker event that proved directory reload."""

    relative = Path(relative_path)
    checkpoint_directory = next(
        (part for part in relative.parts if part.startswith("global_step_")), None,
    )
    if checkpoint_directory is None:
        return {}
    for log in reversed(repository.list_logs(experiment_id)):
        if log.message != "checkpoint.saved":
            continue
        payload = json.loads(log.payload_json)
        event_directory = next(
            (part for part in Path(str(payload.get("path", ""))).parts
             if part.startswith("global_step_")), None,
        )
        if event_directory != checkpoint_directory or payload.get("reload_verified") is not True:
            continue
        evidence: dict[str, Any] = {
            "checkpoint_reload_verified": True,
            "checkpoint_tree_sha256": payload.get("sha256"),
            "checkpoint_directory": checkpoint_directory,
            "verification_source": "gpu-worker-checkpoint-reload",
            "gpu_event_sequence": payload.get("gpu_event_sequence"),
        }
        match = re.fullmatch(r"global_step_(\d+)", checkpoint_directory)
        if match:
            evidence["step"] = int(match.group(1))
        return evidence
    return {}


def _remote_resume_checkpoint(
    repository: PlatformRepository, experiment_id: str, provider_id: str, artifact,
) -> str:
    """Resolve a preserved checkpoint directory without trusting a client path."""

    if provider_id not in {"local-cuda", "runpod-secure-cloud", "modal-sandbox"}:
        raise HTTPException(422, "trainer/provider does not support checkpoint resume")
    metadata = json.loads(artifact.metadata_json)
    relative = Path(str(metadata.get("remote_relative_path", "")))
    if relative.is_absolute() or ".." in relative.parts:
        raise HTTPException(409, "checkpoint lineage contains an unsafe remote path")
    checkpoint_index = next(
        (index for index, part in enumerate(relative.parts) if part.startswith("global_step_")),
        None,
    )
    if checkpoint_index is None:
        raise HTTPException(409, "GPU checkpoint artifact lacks resumable directory lineage")
    if provider_id == "runpod-secure-cloud":
        persistent = any(
            resource.provider == provider_id
            and bool(json.loads(resource.config_json).get("config", {}).get("network_volume_id"))
            for resource in repository.list_compute_resources(experiment_id)
        )
        if not persistent:
            raise HTTPException(409, "RunPod resume requires the original persistent network volume")
    directory = Path(*relative.parts[: checkpoint_index + 1]).as_posix()
    return f"/workspace/runs/{experiment_id}/output/{directory}"


def _resume_storage_binding(
    repository: PlatformRepository, experiment_id: str, provider_id: str,
) -> dict[str, str]:
    """Recover the storage identity that still owns a remote checkpoint."""

    field = {
        "runpod-secure-cloud": ("config", "network_volume_id"),
        "modal-sandbox": ("config", "volume_name"),
        "local-cuda": (None, "data_dir"),
    }.get(provider_id)
    if field is None:
        raise HTTPException(422, "trainer/provider does not support checkpoint resume")
    for resource in repository.list_compute_resources(experiment_id):
        if resource.provider != provider_id:
            continue
        snapshot = json.loads(resource.config_json)
        parent, name = field
        value = snapshot.get(parent, {}).get(name) if parent else snapshot.get(name)
        if isinstance(value, str) and value:
            normalized = str(Path(value).resolve()) if provider_id == "local-cuda" else value
            return {
                "provider_id": provider_id,
                "storage_id": normalized,
                "source_resource_id": resource.id,
            }
    raise HTTPException(409, f"{provider_id} resume requires the original persisted storage")


def _enforce_resume_storage(
    config: dict[str, Any], provider_id: str, selected_storage: str | None,
) -> None:
    binding = config.get("resume_storage")
    if not isinstance(binding, dict):
        return
    if binding.get("provider_id") != provider_id:
        raise HTTPException(409, "resume must use the original execution provider")
    selected = selected_storage or ""
    if provider_id == "local-cuda" and selected:
        selected = str(Path(selected).resolve())
    if not selected or selected != binding.get("storage_id"):
        raise HTTPException(409, "resume must use the original persisted storage")


def _launch_modal(app, repository: PlatformRepository, record, quote_id, confirmation_name):
    if confirmation_name != record.name:
        raise HTTPException(400, "typed confirmation must exactly match the run name")
    if not quote_id:
        raise HTTPException(422, "a persisted Modal quote is required")
    resource = repository.get_compute_resource(quote_id)
    if (
        resource is None or resource.experiment_id != record.id
        or resource.provider != "modal-sandbox" or resource.status != "quoted"
    ):
        raise HTTPException(422, "Modal quote is missing, stale, or belongs to another experiment")
    snapshot = json.loads(resource.config_json)
    if snapshot.get("experiment_config_hash") != record.config_hash:
        raise HTTPException(409, "experiment configuration changed after the quote")
    from .modal import ModalConfig, ModalQuote
    config = ModalConfig(**snapshot["config"])
    reviewed_quote = ModalQuote(**snapshot["quote"])
    provider = _modal_provider(app, snapshot)
    fresh_quote = provider.quote(config)
    if fresh_quote != reviewed_quote:
        snapshot["refreshed_quote"] = asdict(fresh_quote)
        repository.update_compute_resource(resource.id, status="requote_required", config=snapshot)
        raise HTTPException(409, "Modal price or availability changed; review a fresh quote")
    created_sandbox_id: str | None = None
    try:
        repository.transition(record.id, "validating")
        experiment = GPUWorkerExperimentV1.model_validate(snapshot["worker_experiment"])
        staged_path = provider.stage(config, experiment)
        sandbox = provider.provision(
            config, fresh_quote, experiment_path=staged_path,
            confirmation_name=confirmation_name or "",
        )
        external_id = str(sandbox.get("id", ""))
        if not external_id:
            raise RuntimeError("Modal create response did not include a Sandbox id")
        created_sandbox_id = external_id
        snapshot["worker_launch"] = {"experiment_path": staged_path}
        snapshot["provisioned_at"] = datetime.now(UTC).isoformat()
        resource = repository.update_compute_resource(
            resource.id, status="running", external_id=external_id, config=snapshot,
        )
        repository.transition(record.id, "queued", {
            "provider": provider.provider_id, "resource_id": resource.id,
            "external_id": external_id,
        })
    except InvalidTransition as exc:
        if created_sandbox_id:
            try:
                provider.cancel(created_sandbox_id)
            except Exception:
                pass
        raise HTTPException(409, str(exc)) from exc
    except Exception as exc:
        if created_sandbox_id or resource.external_id:
            try:
                provider.cancel(created_sandbox_id or resource.external_id)
            except Exception:
                pass
        try:
            repository.transition(record.id, "failed", {"reason": f"Modal provisioning failed: {exc}"})
        except InvalidTransition:
            pass
        repository.update_compute_resource(resource.id, status="failed")
        raise HTTPException(502, f"Modal provisioning failed: {exc}") from exc
    repository.add_log(record.id, "Modal GPU worker launched", payload={
        "resource_id": resource.id, "external_id": resource.external_id,
        "maximum_cost_usd": reviewed_quote.maximum_cost_usd,
    })
    return {
        "resource": _compute_resource(resource), "preflight": {
            "available": True, "provider_id": provider.provider_id,
            "hardware": {"device_type": reviewed_quote.gpu_type,
                         "device_count": reviewed_quote.gpu_count},
            "cost": {"status": "estimated", "estimated_usd": reviewed_quote.maximum_cost_usd,
                     "basis": snapshot["pricing_basis"]},
        },
    }


def _sync_modal_resource(app, repository: PlatformRepository, record, resource) -> dict[str, Any]:
    snapshot = json.loads(resource.config_json)
    from .modal import ModalConfig, ModalQuote
    config = ModalConfig(**snapshot["config"])
    quote = ModalQuote(**snapshot["quote"])
    provider = _modal_provider(app, snapshot)
    try:
        elapsed_hours = _elapsed_provider_hours(snapshot, resource.created_at)
        stopped = provider.enforce_limits(
            resource.external_id, config, quote, elapsed_hours=elapsed_hours,
        )
        if stopped is not None:
            reason = "Modal maximum runtime or budget reached"
            snapshot["limit_stop"] = {
                "reason": reason, "elapsed_hours": elapsed_hours,
                "stopped_at": datetime.now(UTC).isoformat(),
            }
            _advance_gpu_terminal(repository, record.id, "cancelled", {"reason": reason})
            resource = repository.update_compute_resource(
                resource.id, status="cancelled", config=snapshot,
            )
            repository.add_log(record.id, reason, level="warning", payload={
                "resource_id": resource.id, "elapsed_hours": elapsed_hours,
            })
            return {
                "resource": _compute_resource(resource), "events_synced": 0,
                "artifact_ids": [],
            }
        after = int(snapshot.get("synced_event_sequence", 0))
        events = provider.stream_events(
            resource.external_id, experiment_id=record.id, after_sequence=after,
        )
        terminal = None
        for event in events:
            if event.type == "metric":
                repository.add_metric(
                    record.id, int(event.payload["step"]), str(event.payload["name"]),
                    float(event.payload["value"]), str(event.payload.get("source", "gpu-worker")),
                )
            elif event.type == "log":
                repository.add_log(record.id, str(event.payload.get("message", "")), payload={
                    "gpu_event_sequence": event.sequence, "channel": event.payload.get("channel"),
                })
            else:
                repository.add_log(record.id, event.type, payload={
                    "gpu_event_sequence": event.sequence, **event.payload,
                })
            if event.type in {"worker.completed", "worker.failed", "worker.cancelled"}:
                terminal = event
        if events:
            snapshot["synced_event_sequence"] = events[-1].sequence
        artifacts: list[str] = []
        if terminal and terminal.type == "worker.completed":
            incoming = app.state.data_root / "incoming" / resource.id
            for path in provider.collect_artifacts(config, experiment_id=record.id, destination=incoming):
                artifacts.append(_store_gpu_artifact(app, repository, record, path, incoming))
            _advance_gpu_terminal(repository, record.id, "completed")
            provider.cancel(resource.external_id)
            resource = repository.update_compute_resource(resource.id, status="stopped", config=snapshot)
        elif terminal and terminal.type == "worker.failed":
            _advance_gpu_terminal(repository, record.id, "failed", terminal.payload)
            provider.cancel(resource.external_id)
            resource = repository.update_compute_resource(resource.id, status="failed", config=snapshot)
        elif terminal and terminal.type == "worker.cancelled":
            _advance_gpu_terminal(repository, record.id, "cancelled", terminal.payload)
            provider.cancel(resource.external_id)
            resource = repository.update_compute_resource(resource.id, status="cancelled", config=snapshot)
        else:
            current = repository.get_experiment(record.id)
            if current and current.status == "queued" and events:
                repository.transition(record.id, "running", {"resource_id": resource.id})
            resource = repository.update_compute_resource(resource.id, status="running", config=snapshot)
    except Exception as exc:
        raise HTTPException(502, f"Modal synchronization failed: {exc}") from exc
    return {"resource": _compute_resource(resource), "events_synced": len(events), "artifact_ids": artifacts}


def _elapsed_provider_hours(snapshot: dict[str, Any], fallback: datetime) -> float:
    value = snapshot.get("provisioned_at")
    try:
        started = datetime.fromisoformat(str(value)) if value else fallback
    except ValueError:
        started = fallback
    if started.tzinfo is None:
        started = started.replace(tzinfo=UTC)
    return max(0.0, (datetime.now(UTC) - started.astimezone(UTC)).total_seconds() / 3600)


_SECRET_VALUE_MARKERS = (
    "secret", "password", "credential", "api_key", "api_token",
    "access_token", "auth_token", "private_key",
)
_SECRET_REFERENCE_SUFFIXES = ("secret_name", "secret_names", "secret_reference")


def _is_secret_value_key(key: str) -> bool:
    folded = key.casefold()
    return (
        any(marker in folded for marker in _SECRET_VALUE_MARKERS)
        and not folded.endswith(_SECRET_REFERENCE_SUFFIXES)
    )


def _assert_no_embedded_secrets(value: Any, path: tuple[str, ...] = ()) -> None:
    if isinstance(value, dict):
        for raw_key, item in value.items():
            key = str(raw_key)
            if _is_secret_value_key(key) and item not in (None, "", [], {}):
                location = ".".join((*path, key))
                raise ValueError(
                    f"secret value at {location}; store an environment/provider secret name instead",
                )
            _assert_no_embedded_secrets(item, (*path, key))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_no_embedded_secrets(item, (*path, str(index)))


def _redact_public(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): "[REDACTED]" if _is_secret_value_key(str(key)) else _redact_public(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact_public(item) for item in value]
    return value


def _public_experiment(row) -> dict[str, Any]:
    return _redact_public(serialize_experiment(row))


def _project(row) -> dict[str, Any]:
    return {"id": row.id, "name": row.name, "description": row.description, "created_at": row.created_at.isoformat()}


def _compute_resource(row) -> dict[str, Any]:
    return {
        "id": row.id, "experiment_id": row.experiment_id, "provider": row.provider,
        "external_id": row.external_id, "status": row.status,
        "config": _redact_public(json.loads(row.config_json)),
        "created_at": row.created_at.isoformat(),
    }


def _model(row) -> dict[str, Any]:
    return {
        "id": row.id, "project_id": row.project_id, "display_name": row.display_name,
        "adapter_type": row.adapter_type, "role": row.role, "available": row.available,
        "registration_state": row.registration_state,
        "probe": json.loads(row.probe_json) if row.probe_json else None,
        "config": _redact_public(json.loads(row.config_json)),
    }


async def _read_limited_body(request: Request, maximum_bytes: int) -> bytes:
    supplied = request.headers.get("content-length")
    if supplied:
        try:
            content_length = int(supplied)
        except ValueError as exc:
            raise ValueError("content-length must be an integer") from exc
        if content_length < 0:
            raise ValueError("content-length cannot be negative")
        if content_length > maximum_bytes:
            raise ValueError(f"request body exceeds {maximum_bytes} bytes")
    payload = bytearray()
    async for chunk in request.stream():
        payload.extend(chunk)
        if len(payload) > maximum_bytes:
            raise ValueError(f"request body exceeds {maximum_bytes} bytes")
    return bytes(payload)


def _public_trace_row(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("trace row must be an object")
    if isinstance(value.get("trace"), dict):
        return value
    if value.get("schema_version") != "audioenv.trajectory/v2":
        raise ValueError("unsupported trajectory schema")
    turns = value.get("turns")
    if not isinstance(turns, list) or not turns:
        raise ValueError("native trajectory has no turns")
    terminal = float(value.get("terminal_reward", 0.0))
    process = float(value.get("process_reward", 0.0))
    steps = []
    for index, turn in enumerate(turns, 1):
        if not isinstance(turn, dict) or not isinstance(turn.get("action"), dict):
            raise ValueError("native trajectory turn is invalid")
        final = index == len(turns)
        events = []
        if final and process:
            events.append({"name": "verified_process_reward", "value": process, "category": "process"})
        if final:
            events.append({"name": "terminal_reward", "value": terminal, "category": "terminal"})
        action = turn["action"]
        steps.append({
            "index": index, "action": action,
            "observation": {"reward": terminal + process if final else 0.0,
                            "done": bool(value.get("done")) and final},
            "reward_events": events,
            "state_hash": sha256(json.dumps(action, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            # VeRL V2 currently has no per-turn wall-clock field. Zero is a
            # sentinel paired with the explicit availability flag below.
            "latency_ms": 0.0,
        })
    return {
        "training_step": None,
        "trace": {
            "task_id": str(value.get("task_id", "")), "seed": int(value.get("seed", 0)),
            "audio_profile_id": str((value.get("processor_config") or {}).get("audio_profile_id", "native")),
            "final_metrics": {"terminal_pass": terminal > 0, "terminal_reward": terminal,
                              "process_reward": process, "latency_available": False},
            "steps": steps,
        },
    }


def _restore_trusted_model_images(app: FastAPI, repository: PlatformRepository) -> None:
    """Restore only signed receipts chained to a configured trusted base."""

    pending: list[dict[str, object]] = []
    for artifact in repository.list_artifacts():
        if artifact.kind != "model-conformance-image-receipt":
            continue
        try:
            payload = app.state.artifacts.get_bytes(artifact.id)
            if sha256(payload).hexdigest() != artifact.id:
                continue
            evidence = json.loads(payload)
            receipt = app.state.image_build_authority.verify(evidence)
            metadata = json.loads(artifact.metadata_json)
            if (
                receipt.get("image") != metadata.get("image")
                or receipt.get("base_image") != metadata.get("base_image")
            ):
                continue
            pending.append(receipt)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    changed = True
    while changed:
        changed = False
        for receipt in tuple(pending):
            base = str(receipt.get("base_image", ""))
            image = str(receipt.get("image", ""))
            if base in app.state.trusted_model_conformance_images and re.fullmatch(
                r"(?:[^\s@]+@)?sha256:[0-9a-f]{64}", image,
            ):
                app.state.trusted_model_conformance_images.add(image)
                pending.remove(receipt)
                changed = True


def _environment(row, *, details: bool = False) -> dict[str, Any]:
    data = {
        "id": row.id, "project_id": row.project_id, "version": row.version,
        "content_hash": row.content_hash, "source_kind": row.source_kind,
        "authoring_status": row.authoring_status,
    }
    if details:
        data["package"] = json.loads(row.package_json)
        data["validation"] = json.loads(row.validation_json) if row.validation_json else None
    return data


def _artifact(row) -> dict[str, Any]:
    return {
        "id": row.id, "project_id": row.project_id, "experiment_id": row.experiment_id,
        "kind": row.kind, "media_type": row.media_type, "size_bytes": row.size_bytes,
        "metadata": _redact_public(json.loads(row.metadata_json)),
        "download_url": f"/api/artifacts/{row.id}/download",
    }


def _harness_build(row) -> dict[str, Any]:
    return {
        "id": row.id,
        "project_id": row.project_id,
        "kind": row.kind,
        "source_uri": row.source_uri,
        "status": row.status,
        "report": json.loads(row.report_json),
        "generated_artifact_ids": json.loads(row.generated_artifact_ids_json),
        "environment_id": row.environment_id,
        "model_id": row.model_id,
        "created_at": row.created_at.isoformat(),
    }


def _capability_names(capabilities: dict[str, Any]) -> list[str]:
    names = [key.replace("_", "-") for key, enabled in capabilities.items() if enabled is True]
    checkpoint = capabilities.get("checkpoint_format")
    if checkpoint:
        names.append(str(checkpoint))
    if capabilities.get("distributed"):
        names.append("distributed")
    return sorted(set(names))


def _register_foundry_artifacts(
    repository: PlatformRepository,
    store: LocalArtifactStore,
    artifact_ids: tuple[str, ...],
    *,
    project_id: str | None,
    metadata: dict[str, Any],
) -> None:
    for artifact_id in artifact_ids:
        payload = store.get_bytes(artifact_id)
        media_type = "application/zip" if payload.startswith(b"PK\x03\x04") else "application/json"
        stored = store.describe(artifact_id, media_type=media_type)
        repository.add_artifact(
            stored.id, experiment_id=None, project_id=project_id,
            kind="model-driver-scaffold" if media_type == "application/zip" else "harness-scaffold",
            media_type=stored.media_type,
            size_bytes=stored.size_bytes, path=str(stored.path), metadata=metadata,
        )


def _evaluation(repository: PlatformRepository, row, *, samples: bool = False) -> dict[str, Any]:
    data = {
        "id": row.id, "project_id": row.project_id, "experiment_id": row.experiment_id,
        "checkpoint_id": row.checkpoint_id, "environment_hash": row.environment_hash,
        "verifier_hash": row.verifier_hash, "split": row.split,
        "protocol_hash": row.protocol_hash,
        "config": _redact_public(json.loads(row.config_json)),
        "metrics": json.loads(row.metrics_json),
    }
    if samples:
        data["samples"] = [{
            "task_id": sample.task_id, "audio_profile_id": sample.audio_profile_id,
            "seed": sample.seed, "passed": sample.passed, "reward": sample.reward,
            **json.loads(sample.details_json),
        } for sample in repository.list_evaluation_samples(row.id)]
    return data


def _evaluation_result(repository: PlatformRepository, row):
    protocol = EvaluationProtocolV1.model_validate(json.loads(row.config_json))
    samples = []
    for sample in repository.list_evaluation_samples(row.id):
        details = json.loads(sample.details_json)
        samples.append(EvaluationSampleV1(
            task_id=sample.task_id,
            audio_profile_id=sample.audio_profile_id,
            seed=sample.seed,
            passed=sample.passed,
            reward=sample.reward,
            safety_violations=int(details.get("safety_violations", 0)),
            invalid_calls=int(details.get("invalid_calls", 0)),
            repeated_failed_calls=int(details.get("repeated_failed_calls", 0)),
            failure_category=details.get("failure_category"),
        ))
    result = build_evaluation_result(protocol, samples)
    metrics = json.loads(row.metrics_json)
    expected_rate = metrics.get("success_rate")
    if expected_rate is not None and abs(float(expected_rate) - result.success_rate) > 1e-9:
        raise HTTPException(409, "persisted evaluation aggregate does not match its samples")
    if row.protocol_hash and row.protocol_hash != protocol.protocol_hash:
        raise HTTPException(409, "persisted evaluation protocol hash is invalid")
    return result


def _verified_reward_parity(
    app, repository: PlatformRepository, experiment_id: str,
    artifact_id: str | None, baseline_protocol_hash: str, trained_protocol_hash: str,
) -> bool | None:
    if artifact_id is None:
        return None
    artifact = repository.get_artifact(artifact_id)
    if artifact is None or artifact.experiment_id != experiment_id or artifact.kind != "reward-parity-report":
        raise HTTPException(422, "reward parity artifact is missing or belongs to another experiment")
    try:
        payload = json.loads(app.state.artifacts.get_bytes(artifact_id))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise HTTPException(409, "reward parity artifact is unavailable or invalid") from exc
    if payload.get("schema_version") != "audioenv.reward-parity/v1":
        raise HTTPException(422, "unsupported reward parity evidence schema")
    if payload.get("baseline_protocol_hash") != baseline_protocol_hash:
        raise HTTPException(422, "reward parity evidence is bound to another baseline protocol")
    if payload.get("trained_protocol_hash") != trained_protocol_hash:
        raise HTTPException(422, "reward parity evidence is bound to another trained protocol")
    return bool(payload.get("verified") is True)


app = create_app()
