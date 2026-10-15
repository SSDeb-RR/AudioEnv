"""FastAPI control plane and production SPA host for AudioEnv."""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Literal

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
from ..contracts import GPUExecutionPlanV1, PromotionEvidenceV1
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


def create_app(database_url: str | None = None, *, static_dir: Path | None = None) -> FastAPI:
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
                "supports_quote": True,
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
            return serialize_experiment(repository.request_cancel(experiment_id))
        except KeyError:
            raise HTTPException(404, "experiment not found") from None
        except InvalidTransition as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/experiments/{experiment_id}/launch", status_code=202)
    def launch(experiment_id: str, request: LaunchRequest) -> dict[str, Any]:
        if not request.confirmed:
            raise HTTPException(400, "launch requires explicit confirmation")
        if request.provider != "local-subprocess":
            raise HTTPException(422, "RunPod is blocked until a live quote and paid-run approval are supplied")
        record = repository.get_experiment(experiment_id)
        if record is None:
            raise HTTPException(404, "experiment not found")
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


def _project(row) -> dict[str, Any]:
    return {"id": row.id, "name": row.name, "description": row.description, "created_at": row.created_at.isoformat()}


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
        stored = store.describe(artifact_id, media_type="application/json")
        repository.add_artifact(
            stored.id, experiment_id=None, project_id=project_id,
            kind="harness-scaffold", media_type=stored.media_type,
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
