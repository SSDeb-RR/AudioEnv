"""FastAPI application backed by real AudioEnv platform records."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel, ConfigDict, Field
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Install AudioEnv with the 'platform' extra") from exc

from .db import Database, InvalidTransition, PlatformRepository, serialize_experiment
from ..artifacts import LocalArtifactStore
from ..authoring import CustomTaskInput, DeterministicEnvironmentAuthor


class StrictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExperimentCreate(StrictRequest):
    name: str = Field(min_length=1, max_length=200)
    config: dict[str, Any] = {}


class TransitionRequest(StrictRequest):
    target: str
    payload: dict[str, Any] = {}


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
    config: dict[str, Any] = {}


class LaunchRequest(StrictRequest):
    confirmed: bool
    provider: str = "local-subprocess"


class ModelProbeRequest(StrictRequest):
    model_id: str


def create_app(database_url: str | None = None) -> FastAPI:
    resolved_url = database_url or os.environ.get("AUDIOENV_DATABASE_URL", "sqlite:///./.audioenv/audioenv.db")
    database = Database(resolved_url)
    database.create_schema()
    repository = PlatformRepository(database)
    app = FastAPI(title="AudioEnv", version="0.2.0")
    app.state.database = database
    app.state.repository = repository
    app.state.processes = {}
    if resolved_url.startswith("sqlite:///"):
        artifact_root = Path(resolved_url.removeprefix("sqlite:///")).resolve().parent / "artifacts"
    else:
        artifact_root = Path(os.environ.get("AUDIOENV_ARTIFACT_ROOT", ".audioenv/artifacts")).resolve()
    app.state.artifacts = LocalArtifactStore(artifact_root)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_methods=["GET", "POST"],
        allow_headers=["content-type"],
    )

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": "0.2.0"}

    @app.get("/api/overview")
    def overview() -> dict[str, Any]:
        return repository.overview()

    @app.post("/api/models", status_code=201)
    def register_model(request: ModelCreate) -> dict[str, Any]:
        row = repository.register_model(
            request.id, request.display_name, request.adapter_type, request.role, request.config
        )
        return _model(row)

    @app.get("/api/models")
    def models() -> list[dict[str, Any]]:
        return [_model(row) for row in repository.list_models()]

    @app.post("/api/models/probe")
    def probe_model(request: ModelProbeRequest) -> dict[str, Any]:
        model_id = request.model_id
        row = next((item for item in repository.list_models() if item.id == model_id), None)
        if row is None:
            raise HTTPException(404, "model not found")
        config = json.loads(row.config_json)
        if row.adapter_type == "qwen3-omni":
            from ..models import Qwen3OmniAgentAdapter

            probe = Qwen3OmniAgentAdapter(
                config.get("checkpoint", row.id),
                revision=config.get("revision"),
                allow_download=False,
            ).probe()
        else:
            raise HTTPException(422, f"unsupported model adapter: {row.adapter_type}")
        updated = repository.set_model_probe(model_id, probe.model_dump(mode="json"))
        return _model(updated)

    @app.get("/api/environments")
    def environments() -> list[dict[str, Any]]:
        return [
            {
                "id": row.id,
                "version": row.version,
                "content_hash": row.content_hash,
                "source_kind": row.source_kind,
                "authoring_status": row.authoring_status,
            }
            for row in repository.list_environments()
        ]

    @app.post("/api/environments/custom", status_code=201)
    def create_custom_environment(request: CustomTaskInput) -> dict[str, Any]:
        authored = DeterministicEnvironmentAuthor(app.state.artifacts).author(request)
        report = {
            "passed": authored.validation.passed,
            "checks": [
                {"name": check.name, "passed": check.passed, "details": check.details}
                for check in authored.validation.checks
            ],
        }
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
        return [
            {
                "id": row.id,
                "experiment_id": row.experiment_id,
                "kind": row.kind,
                "media_type": row.media_type,
                "size_bytes": row.size_bytes,
            }
            for row in repository.list_artifacts()
        ]

    @app.get("/api/evaluations")
    def evaluations() -> list[dict[str, Any]]:
        return [
            {
                "id": row.id,
                "experiment_id": row.experiment_id,
                "checkpoint_id": row.checkpoint_id,
                "environment_hash": row.environment_hash,
                "verifier_hash": row.verifier_hash,
                "split": row.split,
                "config": json.loads(row.config_json),
                "metrics": json.loads(row.metrics_json),
            }
            for row in repository.list_evaluations()
        ]

    @app.post("/api/experiments", status_code=201)
    def create_experiment(request: ExperimentCreate) -> dict[str, Any]:
        return serialize_experiment(repository.create_experiment(request.name, request.config))

    @app.get("/api/experiments")
    def list_experiments() -> list[dict[str, Any]]:
        return [serialize_experiment(row) for row in repository.list_experiments()]

    @app.get("/api/experiments/{experiment_id}")
    def get_experiment(experiment_id: str) -> dict[str, Any]:
        record = repository.get_experiment(experiment_id)
        if record is None:
            raise HTTPException(404, "experiment not found")
        return serialize_experiment(record)

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
            raise HTTPException(422, "only the local-subprocess provider is configured")
        record = repository.get_experiment(experiment_id)
        if record is None:
            raise HTTPException(404, "experiment not found")
        config = json.loads(record.config_json)
        if config.get("trainer_id") != "audioenv/tiny-reinforce":
            raise HTTPException(422, "this host only supports the tiny CPU trainer")
        if not resolved_url.startswith("sqlite:///"):
            raise HTTPException(422, "local subprocess launch currently requires SQLite")
        database_path = Path(resolved_url.removeprefix("sqlite:///"))
        from .execution import LocalExecutionProvider
        from ..training import TinyAudioPolicyTrainer, TinyTrainerConfig
        from ..reference import build_reference_package

        trainer_config = TinyTrainerConfig(**config.get("trainer_config", {}))
        preflight = TinyAudioPolicyTrainer(build_reference_package()).preflight(trainer_config)
        if not preflight.available:
            raise HTTPException(422, {"preflight": preflight.model_dump(mode="json")})
        try:
            repository.transition(experiment_id, "validating")
            repository.transition(experiment_id, "queued")
        except InvalidTransition as exc:
            raise HTTPException(409, str(exc)) from exc
        job = repository.create_job(
            experiment_id,
            "tiny_train",
            {
                "config": config.get("trainer_config", {}),
                "output": str(Path(".audioenv/runs") / experiment_id),
            },
        )
        submitted = LocalExecutionProvider().submit(database=database_path, job_id=job.id)
        app.state.processes[job.id] = submitted.process
        return {
            "job_id": job.id,
            "pid": submitted.pid,
            "preflight": preflight.model_dump(mode="json"),
            "cost": {"status": "unavailable", "estimated_usd": None},
        }

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
        return [
            {"id": row.id, "step": row.step, "name": row.name, "value": row.value, "source": row.source}
            for row in repository.list_metrics(experiment_id)
        ]

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
                payload = serialize_experiment(record)
                encoded = json.dumps(payload, separators=(",", ":"))
                if encoded != last:
                    yield f"event: experiment\ndata: {encoded}\n\n"
                    last = encoded
                if record.status in {"completed", "failed", "cancelled"}:
                    return
                await asyncio.sleep(1)

        return StreamingResponse(stream(), media_type="text/event-stream")

    return app


def _model(row) -> dict[str, Any]:
    return {
        "id": row.id,
        "display_name": row.display_name,
        "adapter_type": row.adapter_type,
        "role": row.role,
        "available": row.available,
        "probe": json.loads(row.probe_json) if row.probe_json else None,
        "config": json.loads(row.config_json),
    }


app = create_app()

