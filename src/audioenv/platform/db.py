"""SQLAlchemy persistence for registries, experiments, metrics, and artifacts."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from hashlib import sha256
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

try:
    from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, create_engine, inspect, select, text
    from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker
except ImportError as exc:  # pragma: no cover - exercised by optional-dependency probe
    raise RuntimeError("Install AudioEnv with the 'platform' extra") from exc


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class ProjectRecord(Base):
    __tablename__ = "projects"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, unique=True)
    description: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ModelRecord(Base):
    __tablename__ = "models"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    display_name: Mapped[str] = mapped_column(String)
    adapter_type: Mapped[str] = mapped_column(String)
    role: Mapped[str] = mapped_column(String)
    project_id: Mapped[str | None] = mapped_column(ForeignKey("projects.id"), nullable=True, index=True)
    config_json: Mapped[str] = mapped_column(Text, default="{}")
    probe_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    available: Mapped[bool] = mapped_column(Boolean, default=False)
    registration_state: Mapped[str] = mapped_column(String, default="generated_needs_review")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class EnvironmentRecord(Base):
    __tablename__ = "environments"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    version: Mapped[str] = mapped_column(String)
    content_hash: Mapped[str] = mapped_column(String(64), unique=True)
    source_kind: Mapped[str] = mapped_column(String)
    authoring_status: Mapped[str] = mapped_column(String)
    project_id: Mapped[str | None] = mapped_column(ForeignKey("projects.id"), nullable=True, index=True)
    package_json: Mapped[str] = mapped_column(Text)
    validation_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ExperimentRecord(Base):
    __tablename__ = "experiments"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String)
    status: Mapped[str] = mapped_column(String, index=True)
    project_id: Mapped[str | None] = mapped_column(ForeignKey("projects.id"), nullable=True, index=True)
    config_json: Mapped[str] = mapped_column(Text)
    config_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    preflight_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    environment_id: Mapped[str | None] = mapped_column(ForeignKey("environments.id"), nullable=True)
    agent_model_id: Mapped[str | None] = mapped_column(ForeignKey("models.id"), nullable=True)
    caller_model_id: Mapped[str | None] = mapped_column(ForeignKey("models.id"), nullable=True)
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    failure_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)
    events: Mapped[list["ExperimentEvent"]] = relationship(cascade="all, delete-orphan")
    metrics: Mapped[list["MetricRecord"]] = relationship(cascade="all, delete-orphan")


class ExperimentEvent(Base):
    __tablename__ = "experiment_events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiments.id"), index=True)
    event_type: Mapped[str] = mapped_column(String)
    payload_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class MetricRecord(Base):
    __tablename__ = "metrics"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiments.id"), index=True)
    step: Mapped[int] = mapped_column(Integer)
    name: Mapped[str] = mapped_column(String, index=True)
    value: Mapped[float] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ArtifactRecord(Base):
    __tablename__ = "artifacts"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    experiment_id: Mapped[str | None] = mapped_column(ForeignKey("experiments.id"), nullable=True, index=True)
    project_id: Mapped[str | None] = mapped_column(ForeignKey("projects.id"), nullable=True, index=True)
    kind: Mapped[str] = mapped_column(String)
    media_type: Mapped[str] = mapped_column(String)
    size_bytes: Mapped[int] = mapped_column(Integer)
    path: Mapped[str] = mapped_column(Text)
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class EvaluationRecord(Base):
    __tablename__ = "evaluations"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiments.id"), index=True)
    project_id: Mapped[str | None] = mapped_column(ForeignKey("projects.id"), nullable=True, index=True)
    checkpoint_id: Mapped[str] = mapped_column(String)
    environment_hash: Mapped[str] = mapped_column(String(64))
    verifier_hash: Mapped[str] = mapped_column(String(64))
    split: Mapped[str] = mapped_column(String)
    config_json: Mapped[str] = mapped_column(Text)
    metrics_json: Mapped[str] = mapped_column(Text)
    protocol_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class JobRecord(Base):
    __tablename__ = "jobs"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiments.id"), index=True)
    kind: Mapped[str] = mapped_column(String)
    status: Mapped[str] = mapped_column(String, index=True)
    attempt: Mapped[int] = mapped_column(Integer, default=1)
    worker_id: Mapped[str | None] = mapped_column(String, nullable=True)
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    payload_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class WorkerLogRecord(Base):
    __tablename__ = "worker_logs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiments.id"), index=True)
    level: Mapped[str] = mapped_column(String, default="info")
    message: Mapped[str] = mapped_column(Text)
    payload_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class EvaluationSampleRecord(Base):
    __tablename__ = "evaluation_samples"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    evaluation_id: Mapped[str] = mapped_column(ForeignKey("evaluations.id"), index=True)
    task_id: Mapped[str] = mapped_column(String, index=True)
    audio_profile_id: Mapped[str] = mapped_column(String)
    seed: Mapped[int] = mapped_column(Integer)
    passed: Mapped[bool] = mapped_column(Boolean)
    reward: Mapped[float] = mapped_column(Float)
    details_json: Mapped[str] = mapped_column(Text, default="{}")


class ComparisonRecord(Base):
    __tablename__ = "comparisons"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiments.id"), index=True)
    baseline_evaluation_id: Mapped[str] = mapped_column(ForeignKey("evaluations.id"))
    trained_evaluation_id: Mapped[str] = mapped_column(ForeignKey("evaluations.id"))
    compatibility_hash: Mapped[str] = mapped_column(String(64), index=True)
    result_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class SplitManifestRecord(Base):
    __tablename__ = "split_manifests"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    environment_id: Mapped[str] = mapped_column(String, index=True)
    manifest_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ExternalComputeResourceRecord(Base):
    __tablename__ = "external_compute_resources"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiments.id"), index=True)
    provider: Mapped[str] = mapped_column(String)
    external_id: Mapped[str | None] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(String)
    config_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class HarnessBuildRecord(Base):
    __tablename__ = "harness_builds"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    kind: Mapped[str] = mapped_column(String, index=True)
    source_uri: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String, index=True)
    report_json: Mapped[str] = mapped_column(Text)
    generated_artifact_ids_json: Mapped[str] = mapped_column(Text, default="[]")
    environment_id: Mapped[str | None] = mapped_column(ForeignKey("environments.id"), nullable=True)
    model_id: Mapped[str | None] = mapped_column(ForeignKey("models.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Database:
    def __init__(self, url: str = "sqlite:///./.audioenv/audioenv.db"):
        if url.startswith("sqlite:///"):
            db_path = Path(url.removeprefix("sqlite:///"))
            db_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(url, future=True)
        self.sessions = sessionmaker(self.engine, expire_on_commit=False)

    def create_schema(self) -> None:
        Base.metadata.create_all(self.engine)
        # ``create_all`` does not add columns to databases created by V0.2.
        # Keep local startup self-healing; Alembic remains the authoritative
        # production migration record.
        additions = {
            "models": {
                "project_id": "VARCHAR",
                "registration_state": "VARCHAR DEFAULT 'generated_needs_review'",
            },
            "environments": {"project_id": "VARCHAR"},
            "experiments": {
                "project_id": "VARCHAR", "config_hash": "VARCHAR(64)", "preflight_json": "TEXT",
            },
            "artifacts": {"project_id": "VARCHAR"},
            "evaluations": {"project_id": "VARCHAR", "protocol_hash": "VARCHAR(64)"},
        }
        with self.engine.begin() as connection:
            schema = inspect(connection)
            for table, columns in additions.items():
                existing = {column["name"] for column in schema.get_columns(table)}
                for name, sql_type in columns.items():
                    if name not in existing:
                        connection.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {sql_type}"))

    def session(self) -> Session:
        return self.sessions()


VALID_TRANSITIONS: dict[str, frozenset[str]] = {
    "draft": frozenset({"validating", "cancelled"}),
    "validating": frozenset({"queued", "failed", "cancelled"}),
    "queued": frozenset({"running", "failed", "cancelled"}),
    "running": frozenset({"evaluating", "failed", "cancelled"}),
    "evaluating": frozenset({"completed", "failed", "cancelled"}),
    "completed": frozenset(),
    "failed": frozenset(),
    "cancelled": frozenset(),
}


class InvalidTransition(ValueError):
    pass


class PlatformRepository:
    def __init__(self, database: Database):
        self.database = database

    def ensure_default_project(self) -> ProjectRecord:
        with self.database.session() as session:
            record = session.get(ProjectRecord, "imported-local-workspace")
            if record is None:
                record = ProjectRecord(
                    id="imported-local-workspace",
                    name="Imported local workspace",
                    description="Default project for local and legacy records.",
                )
                session.add(record)
                session.commit()
            return record

    def create_project(self, name: str, description: str = "") -> ProjectRecord:
        record = ProjectRecord(id=str(uuid4()), name=name, description=description)
        with self.database.session() as session:
            session.add(record)
            session.commit()
        return record

    def list_projects(self) -> list[ProjectRecord]:
        with self.database.session() as session:
            return list(session.scalars(select(ProjectRecord).order_by(ProjectRecord.created_at)))

    def get_project(self, project_id: str) -> ProjectRecord | None:
        with self.database.session() as session:
            return session.get(ProjectRecord, project_id)

    def create_experiment(
        self, name: str, config: dict[str, Any], *, project_id: str | None = None
    ) -> ExperimentRecord:
        project = project_id or self.ensure_default_project().id
        canonical = _dump(config)
        record = ExperimentRecord(
            id=str(uuid4()), name=name, status="draft", project_id=project,
            config_json=canonical, config_hash=sha256(canonical.encode()).hexdigest(),
        )
        with self.database.session() as session:
            session.add(record)
            session.add(ExperimentEvent(experiment_id=record.id, event_type="created"))
            session.commit()
        return record

    def register_model(
        self, model_id: str, display_name: str, adapter_type: str, role: str, config: dict[str, Any],
        project_id: str | None = None, registration_state: str = "generated_needs_review",
    ) -> ModelRecord:
        if registration_state not in {"certified", "conformant", "generated_needs_review", "rejected"}:
            raise ValueError("invalid model registration state")
        record = ModelRecord(
            id=model_id,
            display_name=display_name,
            adapter_type=adapter_type,
            role=role,
            project_id=project_id or self.ensure_default_project().id,
            config_json=_dump(config),
            available=False, registration_state=registration_state,
        )
        with self.database.session() as session:
            session.merge(record)
            session.commit()
        return record

    def list_models(self) -> list[ModelRecord]:
        with self.database.session() as session:
            return list(session.scalars(select(ModelRecord).order_by(ModelRecord.created_at.desc())))

    def get_model(self, model_id: str) -> ModelRecord | None:
        with self.database.session() as session:
            return session.get(ModelRecord, model_id)

    def set_model_probe(self, model_id: str, probe: dict[str, Any]) -> ModelRecord:
        with self.database.session() as session:
            row = session.get(ModelRecord, model_id)
            if row is None:
                raise KeyError(model_id)
            row.probe_json = _dump(probe)
            state = str(probe.get("registration_state") or row.registration_state)
            if state not in {"certified", "conformant", "generated_needs_review", "rejected"}:
                raise ValueError("invalid model registration state")
            row.registration_state = state
            row.available = bool(probe.get("available")) and state in {"certified", "conformant"}
            session.commit()
            return row

    def list_environments(self) -> list[EnvironmentRecord]:
        with self.database.session() as session:
            return list(session.scalars(select(EnvironmentRecord).order_by(EnvironmentRecord.created_at.desc())))

    def get_environment(self, environment_id: str) -> EnvironmentRecord | None:
        with self.database.session() as session:
            return session.get(EnvironmentRecord, environment_id)

    def promote_environment(self, environment_id: str, validation: dict[str, Any]) -> EnvironmentRecord:
        with self.database.session() as session:
            record = session.get(EnvironmentRecord, environment_id)
            if record is None:
                raise KeyError(environment_id)
            package = json.loads(record.package_json)
            package["authoring_status"] = "validated"
            package.pop("content_hash", None)
            package["content_hash"] = sha256(
                json.dumps(package, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            record.authoring_status = "validated"
            record.content_hash = package["content_hash"]
            record.package_json = _dump(package)
            existing = json.loads(record.validation_json) if record.validation_json else {}
            if isinstance(existing, dict) and isinstance(existing.get("checks"), list):
                existing["promotion"] = validation
                existing["passed"] = True
                existing.update({key: value for key, value in validation.items() if key != "checks"})
                record.validation_json = _dump(existing)
            else:
                record.validation_json = _dump(validation)
            session.add(record)
            session.commit()
            return record

    def set_environment_validation(
        self, environment_id: str, validation: dict[str, Any],
    ) -> EnvironmentRecord:
        with self.database.session() as session:
            record = session.get(EnvironmentRecord, environment_id)
            if record is None:
                raise KeyError(environment_id)
            record.validation_json = _dump(validation)
            session.add(record)
            session.commit()
            return record

    def add_harness_build(
        self,
        *,
        project_id: str | None,
        kind: str,
        source_uri: str,
        status: str,
        report: dict[str, Any],
        generated_artifact_ids: tuple[str, ...] = (),
        environment_id: str | None = None,
        model_id: str | None = None,
    ) -> HarnessBuildRecord:
        record = HarnessBuildRecord(
            id=str(uuid4()),
            project_id=project_id or self.ensure_default_project().id,
            kind=kind,
            source_uri=source_uri,
            status=status,
            report_json=_dump(report),
            generated_artifact_ids_json=_dump(generated_artifact_ids),
            environment_id=environment_id,
            model_id=model_id,
        )
        with self.database.session() as session:
            session.add(record)
            session.commit()
        return record

    def list_harness_builds(self, kind: str | None = None) -> list[HarnessBuildRecord]:
        with self.database.session() as session:
            query = select(HarnessBuildRecord)
            if kind is not None:
                query = query.where(HarnessBuildRecord.kind == kind)
            return list(session.scalars(query.order_by(HarnessBuildRecord.created_at.desc())))

    def get_harness_build(self, build_id: str) -> HarnessBuildRecord | None:
        with self.database.session() as session:
            return session.get(HarnessBuildRecord, build_id)

    def register_environment(
        self, *, record_id: str, version: str, content_hash: str, source_kind: str,
        authoring_status: str, package: dict[str, Any], validation: dict[str, Any],
        project_id: str | None = None,
    ) -> EnvironmentRecord:
        record = EnvironmentRecord(
            id=record_id, version=version, content_hash=content_hash, source_kind=source_kind,
            authoring_status=authoring_status, package_json=_dump(package), validation_json=_dump(validation),
            project_id=project_id or self.ensure_default_project().id,
        )
        with self.database.session() as session:
            session.merge(record)
            session.commit()
        return record

    def list_artifacts(self) -> list[ArtifactRecord]:
        with self.database.session() as session:
            return list(session.scalars(select(ArtifactRecord).order_by(ArtifactRecord.created_at.desc())))

    def get_artifact(self, artifact_id: str) -> ArtifactRecord | None:
        with self.database.session() as session:
            return session.get(ArtifactRecord, artifact_id)

    def add_artifact(
        self, artifact_id: str, *, experiment_id: str | None, kind: str, media_type: str,
        size_bytes: int, path: str, metadata: dict[str, Any] | None = None,
        project_id: str | None = None,
    ) -> ArtifactRecord:
        record = ArtifactRecord(
            id=artifact_id, experiment_id=experiment_id, kind=kind, media_type=media_type,
            size_bytes=size_bytes, path=path, metadata_json=_dump(metadata or {}),
            project_id=project_id or self.ensure_default_project().id,
        )
        with self.database.session() as session:
            session.merge(record)
            session.commit()
        return record

    def add_evaluation(
        self, *, experiment_id: str, checkpoint_id: str, environment_hash: str,
        verifier_hash: str, split: str, config: dict[str, Any], metrics: dict[str, Any],
        protocol_hash: str | None = None, samples: list[dict[str, Any]] | None = None,
        project_id: str | None = None,
    ) -> EvaluationRecord:
        record = EvaluationRecord(
            id=str(uuid4()), experiment_id=experiment_id, checkpoint_id=checkpoint_id,
            environment_hash=environment_hash, verifier_hash=verifier_hash, split=split,
            config_json=_dump(config), metrics_json=_dump(metrics),
            protocol_hash=protocol_hash,
            project_id=project_id or self.ensure_default_project().id,
        )
        with self.database.session() as session:
            session.add(record)
            for sample in samples or []:
                session.add(EvaluationSampleRecord(
                    evaluation_id=record.id,
                    task_id=str(sample["task_id"]),
                    audio_profile_id=str(sample["audio_profile_id"]),
                    seed=int(sample["seed"]),
                    passed=bool(sample["passed"]),
                    reward=float(sample["reward"]),
                    details_json=_dump({
                        key: value for key, value in sample.items()
                        if key not in {"task_id", "audio_profile_id", "seed", "passed", "reward"}
                    }),
                ))
            session.commit()
        return record

    def get_evaluation(self, evaluation_id: str) -> EvaluationRecord | None:
        with self.database.session() as session:
            return session.get(EvaluationRecord, evaluation_id)

    def list_evaluation_samples(self, evaluation_id: str) -> list[EvaluationSampleRecord]:
        with self.database.session() as session:
            return list(session.scalars(select(EvaluationSampleRecord).where(
                EvaluationSampleRecord.evaluation_id == evaluation_id
            ).order_by(EvaluationSampleRecord.id)))

    def add_comparison(
        self, *, experiment_id: str, baseline_evaluation_id: str,
        trained_evaluation_id: str, compatibility_hash: str, result: dict[str, Any],
    ) -> ComparisonRecord:
        record = ComparisonRecord(
            id=str(uuid4()), experiment_id=experiment_id,
            baseline_evaluation_id=baseline_evaluation_id,
            trained_evaluation_id=trained_evaluation_id,
            compatibility_hash=compatibility_hash, result_json=_dump(result),
        )
        with self.database.session() as session:
            session.add(record)
            session.commit()
        return record

    def list_comparisons(self, experiment_id: str | None = None) -> list[ComparisonRecord]:
        with self.database.session() as session:
            query = select(ComparisonRecord).order_by(ComparisonRecord.created_at.desc())
            if experiment_id:
                query = query.where(ComparisonRecord.experiment_id == experiment_id)
            return list(session.scalars(query))

    def save_split_manifest(self, manifest_id: str, environment_id: str, manifest: dict[str, Any]) -> None:
        with self.database.session() as session:
            session.merge(SplitManifestRecord(id=manifest_id, environment_id=environment_id, manifest_json=_dump(manifest)))
            session.commit()

    def set_preflight(self, experiment_id: str, preflight: dict[str, Any]) -> ExperimentRecord:
        with self.database.session() as session:
            record = session.get(ExperimentRecord, experiment_id)
            if record is None:
                raise KeyError(experiment_id)
            record.preflight_json = _dump(preflight)
            session.commit()
            return record

    def create_compute_resource(
        self, *, experiment_id: str, provider: str, status: str, config: dict[str, Any],
        external_id: str | None = None,
    ) -> ExternalComputeResourceRecord:
        with self.database.session() as session:
            if session.get(ExperimentRecord, experiment_id) is None:
                raise KeyError(experiment_id)
            record = ExternalComputeResourceRecord(
                id=str(uuid4()), experiment_id=experiment_id, provider=provider,
                external_id=external_id, status=status, config_json=_dump(config),
            )
            session.add(record)
            session.commit()
            return record

    def get_compute_resource(self, resource_id: str) -> ExternalComputeResourceRecord | None:
        with self.database.session() as session:
            return session.get(ExternalComputeResourceRecord, resource_id)

    def list_compute_resources(self, experiment_id: str) -> list[ExternalComputeResourceRecord]:
        with self.database.session() as session:
            return list(session.scalars(select(ExternalComputeResourceRecord).where(
                ExternalComputeResourceRecord.experiment_id == experiment_id
            ).order_by(ExternalComputeResourceRecord.created_at.desc())))

    def update_compute_resource(
        self, resource_id: str, *, status: str, external_id: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> ExternalComputeResourceRecord:
        with self.database.session() as session:
            record = session.get(ExternalComputeResourceRecord, resource_id)
            if record is None:
                raise KeyError(resource_id)
            record.status = status
            if external_id is not None:
                record.external_id = external_id
            if config is not None:
                record.config_json = _dump(config)
            session.commit()
            return record

    def add_log(self, experiment_id: str, message: str, *, level: str = "info", payload: dict[str, Any] | None = None) -> WorkerLogRecord:
        record = WorkerLogRecord(experiment_id=experiment_id, level=level, message=message, payload_json=_dump(payload or {}))
        with self.database.session() as session:
            session.add(record)
            session.commit()
        return record

    def list_logs(self, experiment_id: str) -> list[WorkerLogRecord]:
        with self.database.session() as session:
            return list(session.scalars(select(WorkerLogRecord).where(
                WorkerLogRecord.experiment_id == experiment_id
            ).order_by(WorkerLogRecord.id)))

    def list_evaluations(self) -> list[EvaluationRecord]:
        with self.database.session() as session:
            return list(session.scalars(select(EvaluationRecord).order_by(EvaluationRecord.created_at.desc())))

    def list_experiments(self) -> list[ExperimentRecord]:
        with self.database.session() as session:
            return list(session.scalars(select(ExperimentRecord).order_by(ExperimentRecord.created_at.desc())))

    def get_experiment(self, experiment_id: str) -> ExperimentRecord | None:
        with self.database.session() as session:
            return session.get(ExperimentRecord, experiment_id)

    def transition(self, experiment_id: str, target: str, payload: dict[str, Any] | None = None) -> ExperimentRecord:
        with self.database.session() as session:
            record = session.get(ExperimentRecord, experiment_id)
            if record is None:
                raise KeyError(experiment_id)
            if target not in VALID_TRANSITIONS.get(record.status, frozenset()):
                raise InvalidTransition(f"cannot transition {record.status} -> {target}")
            record.status = target
            record.updated_at = utcnow()
            if target == "failed" and payload:
                record.failure_reason = str(payload.get("reason") or "unspecified failure")
            session.add(ExperimentEvent(
                experiment_id=record.id, event_type=f"status:{target}", payload_json=_dump(payload or {})
            ))
            session.commit()
            return record

    def request_cancel(self, experiment_id: str) -> ExperimentRecord:
        with self.database.session() as session:
            record = session.get(ExperimentRecord, experiment_id)
            if record is None:
                raise KeyError(experiment_id)
            if record.status in {"completed", "failed", "cancelled"}:
                raise InvalidTransition(f"cannot cancel {record.status} experiment")
            record.cancel_requested = True
            session.add(ExperimentEvent(experiment_id=record.id, event_type="cancel_requested"))
            session.commit()
            return record

    def add_metric(self, experiment_id: str, step: int, name: str, value: float, source: str) -> MetricRecord:
        with self.database.session() as session:
            if session.get(ExperimentRecord, experiment_id) is None:
                raise KeyError(experiment_id)
            metric = MetricRecord(experiment_id=experiment_id, step=step, name=name, value=value, source=source)
            session.add(metric)
            session.commit()
            return metric

    def create_job(self, experiment_id: str, kind: str, payload: dict[str, Any]) -> JobRecord:
        with self.database.session() as session:
            if session.get(ExperimentRecord, experiment_id) is None:
                raise KeyError(experiment_id)
            job = JobRecord(
                id=str(uuid4()), experiment_id=experiment_id, kind=kind,
                status="queued", payload_json=_dump(payload),
            )
            session.add(job)
            session.commit()
            return job

    def claim_job(self, job_id: str, worker_id: str) -> JobRecord | None:
        """Claim exactly one queued job using a conditional update."""
        from sqlalchemy import update

        with self.database.session() as session:
            result = session.execute(
                update(JobRecord)
                .where(JobRecord.id == job_id, JobRecord.status == "queued")
                .values(status="running", worker_id=worker_id, heartbeat_at=utcnow())
            )
            if result.rowcount != 1:
                session.rollback()
                return None
            session.commit()
            return session.get(JobRecord, job_id)

    def heartbeat_job(self, job_id: str, worker_id: str) -> None:
        from sqlalchemy import update

        with self.database.session() as session:
            result = session.execute(
                update(JobRecord)
                .where(JobRecord.id == job_id, JobRecord.worker_id == worker_id, JobRecord.status == "running")
                .values(heartbeat_at=utcnow())
            )
            if result.rowcount != 1:
                raise KeyError(job_id)
            session.commit()

    def finish_job(self, job_id: str, worker_id: str, status: str) -> JobRecord:
        if status not in {"completed", "failed", "cancelled"}:
            raise ValueError("invalid terminal job status")
        with self.database.session() as session:
            row = session.get(JobRecord, job_id)
            if row is None or row.worker_id != worker_id or row.status != "running":
                raise KeyError(job_id)
            row.status = status
            row.heartbeat_at = utcnow()
            session.commit()
            return row

    def recover_abandoned_jobs(self, stale_after_seconds: int = 60) -> int:
        cutoff = utcnow() - timedelta(seconds=stale_after_seconds)
        with self.database.session() as session:
            jobs = list(session.scalars(
                select(JobRecord).where(JobRecord.status == "running", JobRecord.heartbeat_at < cutoff)
            ))
            for job in jobs:
                job.status = "failed"
                experiment = session.get(ExperimentRecord, job.experiment_id)
                if experiment and experiment.status not in {"completed", "failed", "cancelled"}:
                    experiment.status = "failed"
                    experiment.failure_reason = "worker heartbeat expired"
                    session.add(ExperimentEvent(
                        experiment_id=experiment.id,
                        event_type="status:failed",
                        payload_json=_dump({"reason": "worker heartbeat expired", "job_id": job.id}),
                    ))
            session.commit()
            return len(jobs)

    def list_metrics(self, experiment_id: str) -> list[MetricRecord]:
        with self.database.session() as session:
            query = select(MetricRecord).where(MetricRecord.experiment_id == experiment_id).order_by(MetricRecord.step, MetricRecord.id)
            return list(session.scalars(query))

    def overview(self) -> dict[str, Any]:
        rows = self.list_experiments()
        return {
            "total_experiments": len(rows),
            "active_runs": sum(row.status in {"queued", "running", "evaluating"} for row in rows),
            "failed_runs": sum(row.status == "failed" for row in rows),
            "recent": [serialize_experiment(row) for row in rows[:8]],
            "compute_cost_usd": None,
            "compute_cost_status": "unavailable",
        }


def serialize_experiment(row: ExperimentRecord) -> dict[str, Any]:
    return {
        "id": row.id,
        "name": row.name,
        "status": row.status,
        "project_id": row.project_id,
        "config": json.loads(row.config_json),
        "config_hash": row.config_hash,
        "preflight": json.loads(row.preflight_json) if row.preflight_json else None,
        "environment_id": row.environment_id,
        "agent_model_id": row.agent_model_id,
        "caller_model_id": row.caller_model_id,
        "cancel_requested": row.cancel_requested,
        "failure_reason": row.failure_reason,
        "created_at": row.created_at.isoformat(),
        "updated_at": row.updated_at.isoformat(),
    }


def _dump(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))

