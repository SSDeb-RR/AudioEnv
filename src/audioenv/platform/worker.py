"""Entrypoint for isolated local platform jobs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from hashlib import sha256

from .db import Database, PlatformRepository
from ..reference import build_reference_package
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
        repository.add_log(experiment.id, "Worker claimed training job", payload={"job_id": job.id, "worker_id": worker_id})
        if experiment.status == "queued":
            repository.transition(experiment.id, "running")
        if job.kind not in {"tiny_train", "small_audio_train"}:
            raise ValueError(f"unsupported local job kind: {job.kind}")
        output = Path(payload["output"])
        output.mkdir(parents=True, exist_ok=True)
        trajectory_path = output / "training-trajectories.jsonl"
        config = TinyTrainerConfig(**payload.get("config", {}))
        package = build_reference_package()
        trainer = TinyAudioPolicyTrainer(package)

        def metric(step: int, name: str, value: float, source: str) -> None:
            repository.add_metric(experiment.id, step, name, value, source)
            repository.heartbeat_job(job.id, worker_id)

        def cancelled() -> bool:
            current = repository.get_experiment(experiment.id)
            return bool(current and current.cancel_requested)

        def episode(step: int, trace: dict) -> None:
            row = {"training_step": step, "trace": trace}
            with trajectory_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")

        result = trainer.train(
            config, output, metric_sink=metric, cancel_check=cancelled,
            episode_sink=episode,
        )
        if result.cancelled:
            repository.finish_job(job.id, worker_id, "cancelled")
            repository.transition(experiment.id, "cancelled")
            return 0
        repository.transition(experiment.id, "evaluating")
        repository.add_log(experiment.id, "Checkpoint saved; held-out evaluation started")
        repository.add_metric(experiment.id, result.completed_steps, "baseline_success_rate", result.baseline_success_rate, "evaluator")
        repository.add_metric(experiment.id, result.completed_steps, "trained_success_rate", result.trained_success_rate, "evaluator")
        checkpoint_bytes = result.checkpoint_path.read_bytes()
        checkpoint_id = sha256(checkpoint_bytes).hexdigest()
        repository.add_artifact(
            checkpoint_id, experiment_id=experiment.id, kind="checkpoint",
            media_type="application/x-pytorch", size_bytes=len(checkpoint_bytes),
            path=str(result.checkpoint_path.resolve()),
            metadata={
                "trainer": trainer.trainer_id,
                "claim_scope": "audio_conditioned_platform_proof",
                "checkpoint_reload_verified": result.checkpoint_reload_verified,
                "initial_parameter_hash": result.initial_parameter_hash,
                "final_parameter_hash": result.final_parameter_hash,
            },
            project_id=experiment.project_id,
        )
        trajectory_bytes = trajectory_path.read_bytes()
        repository.add_artifact(
            sha256(trajectory_bytes).hexdigest(), experiment_id=experiment.id,
            kind="training_trajectories", media_type="application/x-ndjson",
            size_bytes=len(trajectory_bytes), path=str(trajectory_path.resolve()),
            metadata={"format": "audioenv.public-trace-jsonl/v1", "observability_included": False,
                      "episode_count": len(trajectory_bytes.splitlines())},
            project_id=experiment.project_id,
        )
        baseline = result.baseline_evaluation
        trained_protocol = result.trained_evaluation.protocol.model_copy(update={"checkpoint_id": checkpoint_id})
        trained = result.trained_evaluation.model_copy(update={"protocol": trained_protocol})
        baseline_row = repository.add_evaluation(
            experiment_id=experiment.id,
            checkpoint_id=baseline.protocol.checkpoint_id,
            environment_hash=baseline.protocol.environment_hash,
            verifier_hash=baseline.protocol.verifier_hash,
            split=baseline.protocol.split,
            config=baseline.protocol.model_dump(mode="json"),
            metrics={
                "success_rate": baseline.success_rate,
                "success_ci_low": baseline.success_ci_low,
                "success_ci_high": baseline.success_ci_high,
                "mean_reward": baseline.mean_reward,
                "failure_counts": baseline.failure_counts,
                "kind": "baseline",
            },
            protocol_hash=baseline.protocol.protocol_hash,
            samples=[sample.model_dump(mode="json") for sample in baseline.samples],
            project_id=experiment.project_id,
        )
        trained_row = repository.add_evaluation(
            experiment_id=experiment.id,
            checkpoint_id=checkpoint_id,
            environment_hash=trained.protocol.environment_hash,
            verifier_hash=trained.protocol.verifier_hash,
            split=trained.protocol.split,
            config=trained.protocol.model_dump(mode="json"),
            metrics={
                "success_rate": trained.success_rate,
                "success_ci_low": trained.success_ci_low,
                "success_ci_high": trained.success_ci_high,
                "mean_reward": trained.mean_reward,
                "failure_counts": trained.failure_counts,
                "kind": "trained",
                "checkpoint_reload_verified": result.checkpoint_reload_verified,
            },
            protocol_hash=trained.protocol.protocol_hash,
            samples=[sample.model_dump(mode="json") for sample in trained.samples],
            project_id=experiment.project_id,
        )
        comparison_payload = result.comparison.model_dump(mode="json")
        # The compatibility result is unchanged when the persisted artifact id
        # replaces the in-memory parameter hash as checkpoint identity.
        comparison_payload["trained_protocol_hash"] = trained.protocol.protocol_hash
        repository.add_comparison(
            experiment_id=experiment.id,
            baseline_evaluation_id=baseline_row.id,
            trained_evaluation_id=trained_row.id,
            compatibility_hash=result.comparison.compatibility_hash,
            result=comparison_payload,
        )
        manifest_path = output / "manifest.json"
        manifest_bytes = manifest_path.read_bytes()
        repository.add_artifact(
            sha256(manifest_bytes).hexdigest(), experiment_id=experiment.id,
            kind="reproducibility_manifest", media_type="application/json",
            size_bytes=len(manifest_bytes), path=str(manifest_path.resolve()),
            metadata={"split_manifest_hash": result.split_manifest_hash},
            project_id=experiment.project_id,
        )
        repository.add_log(
            experiment.id,
            "Training and protocol-compatible held-out comparison completed",
            payload={"checkpoint_reload_verified": result.checkpoint_reload_verified,
                     "paired_delta": result.comparison.paired_delta},
        )
        repository.finish_job(job.id, worker_id, "completed")
        repository.transition(experiment.id, "completed")
        return 0
    except Exception as exc:
        try:
            repository.add_log(experiment.id, "Worker failed", level="error", payload={"reason": str(exc)})
        except Exception:
            pass
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
