"""Provider-neutral AudioEnv worker process.

The worker consumes one immutable experiment bundle and emits versioned JSONL
events. Cloud providers only stage the bundle, launch this command, and relay
its events; they never contain trainer-specific behavior.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from ..contracts import GPUWorkerEventV1, GPUWorkerExperimentV1
from ..reference import build_reference_package
from ..training import (
    TinyAudioPolicyTrainer, TinyTrainerConfig,
    VerlOmniConfig, VerlOmniGRPOTrainer, VerlOmniGSPOTrainer,
)


class EventWriter:
    def __init__(self, experiment_id: str, destination: Path | None = None):
        self.experiment_id, self.destination, self.sequence = experiment_id, destination, 0
        if destination:
            destination.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event_type: str, **payload: Any) -> GPUWorkerEventV1:
        self.sequence += 1
        event = GPUWorkerEventV1(
            sequence=self.sequence, experiment_id=self.experiment_id, type=event_type,
            timestamp=datetime.now(UTC).isoformat(), payload=payload,
        )
        line = event.model_dump_json()
        print(line, flush=True)
        if self.destination:
            with self.destination.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
        return event


def load_experiment(path: Path) -> GPUWorkerExperimentV1:
    return GPUWorkerExperimentV1.model_validate_json(path.read_text(encoding="utf-8"))


def run_experiment(experiment: GPUWorkerExperimentV1) -> int:
    output = Path(experiment.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    events = EventWriter(experiment.experiment_id, output / "events.jsonl")
    events.emit(
        "worker.started", worker_version="0.6.0-dev", pid=os.getpid(),
        experiment_hash=experiment.content_hash, backend=experiment.backend,
    )
    try:
        if experiment.backend == "tiny-cpu-proof":
            result = _run_tiny(experiment, output, events)
        else:
            result = _run_qwen(experiment, output, events)
        if result == 130:
            events.emit("worker.cancelled", reason="cancellation requested")
            return result
        artifacts = _artifact_manifest(output)
        manifest = output / "worker-artifacts.json"
        manifest.write_text(json.dumps(artifacts, sort_keys=True, indent=2), encoding="utf-8")
        events.emit("artifact.manifest", path=str(manifest), artifacts=artifacts)
        events.emit("worker.completed", artifact_count=len(artifacts))
        return 0
    except Exception as exc:
        events.emit("worker.failed", error_type=type(exc).__name__, reason=str(exc))
        return 1


def _run_tiny(experiment: GPUWorkerExperimentV1, output: Path, events: EventWriter) -> int:
    config = TinyTrainerConfig(**experiment.hyperparameters)
    trainer = TinyAudioPolicyTrainer(build_reference_package())
    estimate = trainer.preflight(config)
    events.emit("preflight.completed", available=estimate.available, report=estimate.model_dump(mode="json"))
    if not estimate.available:
        raise RuntimeError("local trainer preflight failed: " + "; ".join(estimate.reasons))
    events.emit("training.started", trainer=trainer.trainer_id, algorithm="reinforce")

    def cancelled() -> bool:
        return bool(experiment.cancellation_file and Path(experiment.cancellation_file).exists())

    def metric(step: int, name: str, value: float, source: str) -> None:
        events.emit("metric", step=step, name=name, value=value, source=source)

    result = trainer.train(config, output, metric_sink=metric, cancel_check=cancelled)
    if result.cancelled:
        return 130
    checkpoint_hash = sha256(result.checkpoint_path.read_bytes()).hexdigest()
    events.emit(
        "checkpoint.saved", path=str(result.checkpoint_path), sha256=checkpoint_hash,
        reload_verified=result.checkpoint_reload_verified,
        initial_parameter_hash=result.initial_parameter_hash,
        final_parameter_hash=result.final_parameter_hash,
    )
    events.emit(
        "evaluation.completed", baseline_success_rate=result.baseline_success_rate,
        trained_success_rate=result.trained_success_rate,
        paired_delta=result.comparison.paired_delta,
        compatibility_hash=result.comparison.compatibility_hash,
    )
    return 0


def _run_qwen(experiment: GPUWorkerExperimentV1, output: Path, events: EventWriter) -> int:
    trainer = VerlOmniGSPOTrainer() if experiment.algorithm == "gspo" else VerlOmniGRPOTrainer()
    hp = experiment.hyperparameters
    config = VerlOmniConfig(
        checkout=Path(experiment.verl_checkout or ""), model_path=Path(experiment.model_path or ""),
        caller_model_path=Path(experiment.caller_model_path or ""),
        dataset_path=Path(experiment.rollout_dataset_path or ""),
        environment_manifest=Path(experiment.environment_manifest or ""), output_dir=output,
        algorithm=experiment.algorithm, nodes=int(hp.get("nodes", 1)),
        gpus_per_node=int(hp.get("gpus_per_node", 4)),
        learning_rate=float(hp.get("learning_rate", 1e-6)),
        lora_rank=int(hp.get("lora_rank", 32)), lora_alpha=int(hp.get("lora_alpha", 64)),
        group_size=int(hp.get("group_size", 4)), max_steps=int(hp.get("max_steps", 1)),
        resume_from=Path(experiment.resume_from) if experiment.resume_from else None,
    )
    estimate = trainer.preflight(config)
    events.emit("preflight.completed", available=estimate.available, report=estimate.model_dump(mode="json"))
    if not estimate.available:
        raise RuntimeError("Qwen trainer preflight failed: " + "; ".join(estimate.reasons))
    command = trainer.launch_command(config)
    events.emit("training.started", trainer=trainer.trainer_id, algorithm=experiment.algorithm, command=list(command))
    stdout_path, stderr_path = output / "trainer.stdout.log", output / "trainer.stderr.log"
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(command, cwd=config.checkout, env=trainer.launch_environment(config), stdout=stdout, stderr=stderr)
        while process.poll() is None:
            if experiment.cancellation_file and Path(experiment.cancellation_file).exists():
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()
                return 130
            time.sleep(0.5)
    if process.returncode:
        raise RuntimeError(f"VeRL-Omni exited with status {process.returncode}; inspect trainer logs")
    return 0


def _artifact_manifest(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.name != "worker-artifacts.json"):
        payload = path.read_bytes()
        rows.append({"path": str(path.relative_to(root)), "size_bytes": len(payload), "sha256": sha256(payload).hexdigest()})
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="audioenv-gpu-worker")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("experiment", type=Path)
    args = parser.parse_args(argv)
    return run_experiment(load_experiment(args.experiment))


if __name__ == "__main__":
    raise SystemExit(main())
