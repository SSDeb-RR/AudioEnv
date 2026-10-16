"""FastAPI control plane and production SPA host for AudioEnv."""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Callable, Literal

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, StreamingResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel, ConfigDict, Field
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Install AudioEnv with the 'platform' extra") from exc

from .db import Database, InvalidTransition, PlatformRepository, serialize_experiment
from ..artifacts import LocalArtifactStore
from ..authoring import CustomTaskInput, DeterministicEnvironmentAuthor
from ..evaluation import package_hashes
from ..contracts import GPUExecutionPlanV1, GPUWorkerExperimentV1, PromotionEvidenceV1
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


class ModelFoundryRequest(StrictRequest):
    project_id: str | None = None
    model_id: str = Field(min_length=1, max_length=500)
    source: str = Field(min_length=1, max_length=4_096)
    role: str = Field(default="agent", pattern="^(agent|caller|both)$")


class CompatibilityRequest(StrictRequest):
    environment_id: str
    model_id: str
    caller_id: str | None = None


def create_app(
    database_url: str | None = None, *, static_dir: Path | None = None,
    runpod_transport: Any | None = None,
    runpod_executor_factory: Callable[[dict[str, Any], Any], Any] | None = None,
    modal_transport: Any | None = None,
    local_cuda_provider: Any | None = None,
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
    app.state.promotion_authority = PromotionEvidenceAuthority(data_root / "secrets" / "promotion-evidence.key")
    app.state.harness_registry = build_builtin_registry()
    app.state.environment_foundry = EnvironmentFoundry(data_root / "foundry" / "sources", app.state.artifacts)
    app.state.model_foundry = ModelFoundry(app.state.harness_registry, app.state.artifacts)
    app.state.validation_runner = IsolatedValidationRunner()
    app.state.runpod_transport = runpod_transport
    app.state.runpod_executor_factory = runpod_executor_factory
    app.state.modal_transport = modal_transport
    app.state.local_cuda_provider = local_cuda_provider
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
        return repository.overview()

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
            {"checkpoint": request.source, "onboarding": report.model_dump(mode="json")}, request.project_id,
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
        probe["conformance"] = report.model_dump(mode="json")
        repository.set_model_probe(model_id, probe)
        return report.model_dump(mode="json")

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
        return FileResponse(path, media_type=row.media_type, filename=path.name)

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

    @app.post("/api/experiments", status_code=201)
    def create_experiment(request: ExperimentCreate) -> dict[str, Any]:
        if request.project_id and repository.get_project(request.project_id) is None:
            raise HTTPException(422, "project not found")
        return serialize_experiment(repository.create_experiment(
            request.name, request.config, project_id=request.project_id,
        ))

    @app.get("/api/experiments")
    def list_experiments() -> list[dict[str, Any]]:
        return [serialize_experiment(row) for row in repository.list_experiments()]

    @app.get("/api/experiments/{experiment_id}")
    def get_experiment(experiment_id: str) -> dict[str, Any]:
        record = repository.get_experiment(experiment_id)
        if record is None:
            raise HTTPException(404, "experiment not found")
        return serialize_experiment(record)

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
            return serialize_experiment(repository.transition(experiment_id, request.target, request.payload))
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
        return serialize_experiment(record)

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
            "running": "provisioning", "stopped": "stopped", "exited": "stopped",
            "terminated": "stopped", "created": "provisioning", "pending": "provisioning",
        }.get(remote_status, resource.status)
        snapshot = json.loads(resource.config_json)
        if remote_status == "running" and "worker_launch" in snapshot:
            mapped = "running"
        snapshot["last_remote_state"] = {
            key: remote[key] for key in (
                "id", "status", "desiredStatus", "publicIp", "ssh_host", "sshHost",
                "ssh_port", "sshPort",
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
        from .runpod import RunPodConfig, RunPodRESTV2Transport, RunPodSecureCloudProvider
        runpod_config = RunPodConfig(**snapshot["config"])
        provider = RunPodSecureCloudProvider(
            app.state.runpod_transport
            or RunPodRESTV2Transport(api_key_env=runpod_config.api_key_secret_name)
        )
        try:
            pod = provider.reconnect(resource.external_id)
            executor = _resolve_runpod_executor(app, pod, runpod_config, snapshot)
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
                    payload = path.read_bytes()
                    media_type = "application/json" if path.suffix in {".json", ".jsonl"} else "application/octet-stream"
                    stored = app.state.artifacts.put_bytes(payload, media_type=media_type)
                    kind = "checkpoint" if "checkpoint" in path.parts or "adapter" in path.name else "gpu_run_artifact"
                    row = repository.add_artifact(
                        stored.id, experiment_id=experiment_id, project_id=record.project_id,
                        kind=kind, media_type=media_type, size_bytes=stored.size_bytes,
                        path=str(stored.path), metadata={"remote_relative_path": str(path.relative_to(incoming))},
                    )
                    artifacts.append(row.id)
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
            "payload": json.loads(row.payload_json), "created_at": row.created_at.isoformat(),
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
        try:
            model, payload = TinyAudioPolicyTrainer(build_reference_package()).load_checkpoint(Path(row.path))
            del model
            return {"verified": True, "artifact_id": artifact_id, "step": int(payload["step"])}
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
                encoded = json.dumps(serialize_experiment(record), separators=(",", ":"))
                if encoded != last:
                    yield f"event: experiment\ndata: {encoded}\n\n"
                    last = encoded
                if record.status in {"completed", "failed", "cancelled"}:
                    return
                await asyncio.sleep(1)

        return StreamingResponse(stream(), media_type="text/event-stream")

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
            {"claim_scope": "audio_conditioned_platform_proof"},
        )
        estimate = TinyAudioPolicyTrainer(package).preflight(TinyTrainerConfig(steps=1))
        repository.set_model_probe(model_id, {
            "available": estimate.available, "device": "cpu", "revision": "builtin-v2",
            "reasons": list(estimate.reasons), "capabilities": builtin_capabilities,
        })
    else:
        existing = repository.get_model(model_id)
        probe = json.loads(existing.probe_json) if existing and existing.probe_json else {}
        if "capabilities" not in probe:
            probe["capabilities"] = builtin_capabilities
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
    host = pod.get("ssh_host", pod.get("sshHost", pod.get("publicIp")))
    port = int(pod.get("ssh_port", pod.get("sshPort", 22)))
    key_value = os.environ.get(runpod_config.ssh_key_secret_name)
    known_value = os.environ.get(snapshot.get("known_hosts_secret_name", "RUNPOD_KNOWN_HOSTS"))
    if not host or not key_value or not known_value:
        return None
    return OpenSSHRemoteExecutor(
        host=str(host), port=port, key_path=Path(key_value),
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
        resource = repository.update_compute_resource(
            resource.id, status="provisioning", external_id=external_id,
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
    kind = "checkpoint" if "checkpoint" in path.parts or "adapter" in path.name else "gpu_run_artifact"
    row = repository.add_artifact(
        stored.id, experiment_id=record.id, project_id=record.project_id,
        kind=kind, media_type=media_type, size_bytes=stored.size_bytes,
        path=str(stored.path), metadata={"remote_relative_path": str(path.relative_to(incoming))},
    )
    return row.id


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
    from .modal import ModalConfig
    config = ModalConfig(**snapshot["config"])
    provider = _modal_provider(app, snapshot)
    try:
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
                payload = path.read_bytes()
                media_type = "application/json" if path.suffix in {".json", ".jsonl"} else "application/octet-stream"
                stored = app.state.artifacts.put_bytes(payload, media_type=media_type)
                kind = "checkpoint" if "checkpoint" in path.parts or "adapter" in path.name else "gpu_run_artifact"
                row = repository.add_artifact(
                    stored.id, experiment_id=record.id, project_id=record.project_id,
                    kind=kind, media_type=media_type, size_bytes=stored.size_bytes,
                    path=str(stored.path), metadata={"remote_relative_path": str(path.relative_to(incoming))},
                )
                artifacts.append(row.id)
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


def _project(row) -> dict[str, Any]:
    return {"id": row.id, "name": row.name, "description": row.description, "created_at": row.created_at.isoformat()}


def _compute_resource(row) -> dict[str, Any]:
    return {
        "id": row.id, "experiment_id": row.experiment_id, "provider": row.provider,
        "external_id": row.external_id, "status": row.status,
        "config": json.loads(row.config_json), "created_at": row.created_at.isoformat(),
    }


def _model(row) -> dict[str, Any]:
    return {
        "id": row.id, "project_id": row.project_id, "display_name": row.display_name,
        "adapter_type": row.adapter_type, "role": row.role, "available": row.available,
        "probe": json.loads(row.probe_json) if row.probe_json else None,
        "config": json.loads(row.config_json),
    }


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
        "metadata": json.loads(row.metadata_json), "download_url": f"/api/artifacts/{row.id}/download",
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
        "protocol_hash": row.protocol_hash, "config": json.loads(row.config_json),
        "metrics": json.loads(row.metrics_json),
    }
    if samples:
        data["samples"] = [{
            "task_id": sample.task_id, "audio_profile_id": sample.audio_profile_id,
            "seed": sample.seed, "passed": sample.passed, "reward": sample.reward,
            **json.loads(sample.details_json),
        } for sample in repository.list_evaluation_samples(row.id)]
    return data


app = create_app()
