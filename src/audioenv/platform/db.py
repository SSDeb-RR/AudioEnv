"""SQLAlchemy persistence for registries, experiments, metrics, and artifacts."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

try:
    from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, create_engine, select
    from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker
except ImportError as exc:  # pragma: no cover - exercised by optional-dependency probe
    raise RuntimeError("Install AudioEnv with the 'platform' extra") from exc


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class ModelRecord(Base):
    __tablename__ = "models"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    display_name: Mapped[str] = mapped_column(String)
    adapter_type: Mapped[str] = mapped_column(String)
    role: Mapped[str] = mapped_column(String)
    config_json: Mapped[str] = mapped_column(Text, default="{}")
    probe_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    available: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class EnvironmentRecord(Base):
    __tablename__ = "environments"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    version: Mapped[str] = mapped_column(String)
    content_hash: Mapped[str] = mapped_column(String(64), unique=True)
    source_kind: Mapped[str] = mapped_column(String)
    authoring_status: Mapped[str] = mapped_column(String)
    package_json: Mapped[str] = mapped_column(Text)
    validation_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ExperimentRecord(Base):
    __tablename__ = "experiments"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String)
    status: Mapped[str] = mapped_column(String, index=True)
    config_json: Mapped[str] = mapped_column(Text)
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
    checkpoint_id: Mapped[str] = mapped_column(String)
    environment_hash: Mapped[str] = mapped_column(String(64))
    verifier_hash: Mapped[str] = mapped_column(String(64))
    split: Mapped[str] = mapped_column(String)
    config_json: Mapped[str] = mapped_column(Text)
    metrics_json: Mapped[str] = mapped_column(Text)
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


class Database:
    def __init__(self, url: str = "sqlite:///./.audioenv/audioenv.db"):
        if url.startswith("sqlite:///"):
            db_path = Path(url.removeprefix("sqlite:///"))
            db_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(url, future=True)
        self.sessions = sessionmaker(self.engine, expire_on_commit=False)

    def create_schema(self) -> None:
        Base.metadata.create_all(self.engine)

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

    def create_experiment(self, name: str, config: dict[str, Any]) -> ExperimentRecord:
        record = ExperimentRecord(
            id=str(uuid4()), name=name, status="draft", config_json=_dump(config)
        )
        with self.database.session() as session:
            session.add(record)
            session.add(ExperimentEvent(experiment_id=record.id, event_type="created"))
            session.commit()
        return record

    def register_model(
        self, model_id: str, display_name: str, adapter_type: str, role: str, config: dict[str, Any]
    ) -> ModelRecord:
        record = ModelRecord(
            id=model_id,
            display_name=display_name,
            adapter_type=adapter_type,
            role=role,
            config_json=_dump(config),
            available=False,
        )
        with self.database.session() as session:
            session.merge(record)
            session.commit()
        return record

    def list_models(self) -> list[ModelRecord]:
        with self.database.session() as session:
            return list(session.scalars(select(ModelRecord).order_by(ModelRecord.created_at.desc())))

    def set_model_probe(self, model_id: str, probe: dict[str, Any]) -> ModelRecord:
        with self.database.session() as session:
            row = session.get(ModelRecord, model_id)
            if row is None:
                raise KeyError(model_id)
            row.probe_json = _dump(probe)
            row.available = bool(probe.get("available"))
            session.commit()
            return row

    def list_environments(self) -> list[EnvironmentRecord]:
        with self.database.session() as session:
            return list(session.scalars(select(EnvironmentRecord).order_by(EnvironmentRecord.created_at.desc())))

    def register_environment(
        self, *, record_id: str, version: str, content_hash: str, source_kind: str,
        authoring_status: str, package: dict[str, Any], validation: dict[str, Any],
    ) -> EnvironmentRecord:
        record = EnvironmentRecord(
            id=record_id, version=version, content_hash=content_hash, source_kind=source_kind,
            authoring_status=authoring_status, package_json=_dump(package), validation_json=_dump(validation),
        )
        with self.database.session() as session:
            session.merge(record)
            session.commit()
        return record

    def list_artifacts(self) -> list[ArtifactRecord]:
        with self.database.session() as session:
            return list(session.scalars(select(ArtifactRecord).order_by(ArtifactRecord.created_at.desc())))

    def add_artifact(
        self, artifact_id: str, *, experiment_id: str | None, kind: str, media_type: str,
        size_bytes: int, path: str, metadata: dict[str, Any] | None = None,
    ) -> ArtifactRecord:
        record = ArtifactRecord(
            id=artifact_id, experiment_id=experiment_id, kind=kind, media_type=media_type,
            size_bytes=size_bytes, path=path, metadata_json=_dump(metadata or {}),
        )
        with self.database.session() as session:
            session.merge(record)
            session.commit()
        return record

    def add_evaluation(
        self, *, experiment_id: str, checkpoint_id: str, environment_hash: str,
        verifier_hash: str, split: str, config: dict[str, Any], metrics: dict[str, Any],
    ) -> EvaluationRecord:
        record = EvaluationRecord(
            id=str(uuid4()), experiment_id=experiment_id, checkpoint_id=checkpoint_id,
            environment_hash=environment_hash, verifier_hash=verifier_hash, split=split,
            config_json=_dump(config), metrics_json=_dump(metrics),
        )
        with self.database.session() as session:
            session.add(record)
            session.commit()
        return record

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
        "config": json.loads(row.config_json),
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

