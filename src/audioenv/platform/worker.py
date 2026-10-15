"""Entrypoint for isolated local platform jobs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from hashlib import sha256

from .db import Database, PlatformRepository
from ..reference import build_reference_package
from ..evaluation import package_hashes
from ..training import TinyAudioPolicyTrainer, TinyTrainerConfig


def run_job(database_path: Path, job_id: str) -> int:
    database = Database(f"sqlite:///{database_path}")
    repository = PlatformRepository(database)
    worker_id = f"local-{os.getpid()}"
    job = repository.claim_job(job_id, worker_id)
    if job is None:
        return 2
    experiment = repository.get_experiment(job.experiment_id)
    if experiment is None:
        repository.finish_job(job.id, worker_id, "failed")
        return 2
    payload = json.loads(job.payload_json)
    try:
        if experiment.status == "queued":
            repository.transition(experiment.id, "running")
        if job.kind != "tiny_train":
            raise ValueError(f"unsupported local job kind: {job.kind}")
        output = Path(payload["output"])
        config = TinyTrainerConfig(**payload.get("config", {}))
        package = build_reference_package()
        trainer = TinyAudioPolicyTrainer(package)

        def metric(step: int, name: str, value: float, source: str) -> None:
            repository.add_metric(experiment.id, step, name, value, source)
            repository.heartbeat_job(job.id, worker_id)

        def cancelled() -> bool:
            current = repository.get_experiment(experiment.id)
            return bool(current and current.cancel_requested)

        result = trainer.train(config, output, metric_sink=metric, cancel_check=cancelled)
        if result.cancelled:
            repository.finish_job(job.id, worker_id, "cancelled")
            repository.transition(experiment.id, "cancelled")
            return 0
        repository.transition(experiment.id, "evaluating")
        repository.add_metric(experiment.id, result.completed_steps, "baseline_success_rate", result.baseline_success_rate, "evaluator")
        repository.add_metric(experiment.id, result.completed_steps, "trained_success_rate", result.trained_success_rate, "evaluator")
        checkpoint_bytes = result.checkpoint_path.read_bytes()
        checkpoint_id = sha256(checkpoint_bytes).hexdigest()
        repository.add_artifact(
            checkpoint_id, experiment_id=experiment.id, kind="checkpoint",
            media_type="application/x-pytorch", size_bytes=len(checkpoint_bytes),
            path=str(result.checkpoint_path.resolve()),
            metadata={"trainer": trainer.trainer_id, "mechanics_only": True},
        )
        environment_hash, verifier_hash = package_hashes(package)
        repository.add_evaluation(
            experiment_id=experiment.id,
            checkpoint_id=checkpoint_id,
            environment_hash=environment_hash,
            verifier_hash=verifier_hash,
            split="fixture",
            config={"seeds": [10_000, 10_001], "independent_audio": False},
            metrics={
                "baseline_success_rate": result.baseline_success_rate,
                "trained_success_rate": result.trained_success_rate,
                "learning_lift": result.trained_success_rate - result.baseline_success_rate,
                "claim_scope": "cpu_mechanics_proof",
            },
        )
        repository.finish_job(job.id, worker_id, "completed")
        repository.transition(experiment.id, "completed")
        return 0
    except Exception as exc:
        repository.finish_job(job.id, worker_id, "failed")
        current = repository.get_experiment(experiment.id)
        if current and current.status not in {"failed", "completed", "cancelled"}:
            repository.transition(experiment.id, "failed", {"reason": str(exc)})
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args(argv)
    return run_job(args.database, args.job_id)


if __name__ == "__main__":
    raise SystemExit(main())
