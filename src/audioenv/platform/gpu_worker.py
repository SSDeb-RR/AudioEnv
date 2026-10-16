"""Provider-neutral AudioEnv worker process.

The worker consumes one immutable experiment bundle and emits versioned JSONL
events. Cloud providers only stage the bundle, launch this command, and relay
its events; they never contain trainer-specific behavior.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import selectors
import subprocess
import sys
import time
from typing import Any
from urllib.request import urlopen

from ..contracts import GPUWorkerEventV1, GPUWorkerExperimentV1
from ..reference import build_reference_package
from .gpu_runtime import attest_qwen_runtime
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
    (output / "experiment.snapshot.json").write_text(
        experiment.model_dump_json(indent=2), encoding="utf-8",
    )
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
    runtime = attest_qwen_runtime(experiment)
    events.emit(
        "preflight.completed", available=runtime.available,
        report={"scope": "immutable-runtime", "reasons": runtime.reasons, "evidence": runtime.evidence},
    )
    if not runtime.available:
        raise RuntimeError("GPU runtime attestation failed: " + "; ".join(runtime.reasons))
    trainer = VerlOmniGSPOTrainer() if experiment.algorithm == "gspo" else VerlOmniGRPOTrainer()
    hp = experiment.hyperparameters
    config = VerlOmniConfig(
        checkout=Path(experiment.verl_checkout or ""), model_path=Path(experiment.model_path or ""),
        caller_model_path=Path(experiment.caller_model_path or ""),
        dataset_path=Path(experiment.rollout_dataset_path or ""),
        validation_dataset_path=Path(experiment.validation_dataset_path or ""),
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
    allocation = QwenGPUAllocation.from_hyperparameters(hp)
    caller_port = int(hp.get("caller_port", 8788))
    caller_url = f"http://127.0.0.1:{caller_port}"
    caller_stdout = (output / "caller.stdout.log").open("w", encoding="utf-8")
    caller_stderr = (output / "caller.stderr.log").open("w", encoding="utf-8")
    caller = subprocess.Popen(
        ["audioenv-qwen-caller", "--checkpoint", str(config.caller_model_path), "--port", str(caller_port)],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": allocation.caller},
        stdout=caller_stdout, stderr=caller_stderr,
    )
    try:
        _wait_for_caller(caller, caller_url, timeout_seconds=float(hp.get("caller_startup_timeout", 900)))
        events.emit(
            "service.started", service="frozen-qwen-caller", url=caller_url,
            cuda_devices=allocation.caller_devices,
        )
        command = trainer.launch_command(config)
        events.emit("training.started", trainer=trainer.trainer_id, algorithm=experiment.algorithm, command=list(command))
        stdout_path, stderr_path = output / "trainer.stdout.log", output / "trainer.stderr.log"
        train_environment = trainer.launch_environment(config)
        train_environment.update({
            "CUDA_VISIBLE_DEVICES": allocation.trainer,
            "AUDIOENV_CALLER_URL": caller_url,
            "AUDIOENV_TAU2_CHECKOUT": str(Path(experiment.tau2_checkout or "").resolve()),
            "AUDIOENV_ROLLOUT_AUDIO_DIR": str((output / "rollout-audio").resolve()),
        })
        process = subprocess.Popen(
            command, cwd=config.checkout, env=train_environment,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1,
        )
        cancelled = _relay_trainer_process(
            process, caller, experiment, events,
            stdout_path=stdout_path, stderr_path=stderr_path,
        )
        if cancelled:
            return 130
        if process.returncode:
            raise RuntimeError(f"VeRL-Omni exited with status {process.returncode}; inspect trainer logs")
        checkpoint = _latest_checkpoint(output)
        if checkpoint is not None:
            events.emit(
                "checkpoint.saved", path=str(checkpoint), sha256=_hash_path(checkpoint),
                reload_verified=False, reason="reload verification requires the evaluation gate",
            )
        return 0
    finally:
        if caller.poll() is None:
            caller.terminate()
            try:
                caller.wait(timeout=30)
            except subprocess.TimeoutExpired:
                caller.kill(); caller.wait()
        caller_stdout.close(); caller_stderr.close()


@dataclass(frozen=True, slots=True)
class QwenGPUAllocation:
    trainer_devices: tuple[int, ...]
    caller_devices: tuple[int, ...]

    @property
    def trainer(self) -> str:
        return ",".join(str(item) for item in self.trainer_devices)

    @property
    def caller(self) -> str:
        return ",".join(str(item) for item in self.caller_devices)

    @classmethod
    def from_hyperparameters(cls, values: dict[str, Any]) -> "QwenGPUAllocation":
        trainer = _device_list(values.get("trainer_cuda_devices", "0,1,2,3"))
        caller = _device_list(values.get("caller_cuda_devices", "4,5"))
        if len(trainer) < 4 or len(caller) < 1:
            raise ValueError("Qwen GPU run requires at least four trainer GPUs and one caller GPU")
        if set(trainer) & set(caller):
            raise ValueError("trainer and frozen caller GPU allocations must be disjoint")
        return cls(trainer, caller)


def _device_list(value: object) -> tuple[int, ...]:
    try:
        devices = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
    except ValueError as exc:
        raise ValueError("CUDA device allocations must be comma-separated integers") from exc
    if not devices or len(set(devices)) != len(devices) or any(item < 0 for item in devices):
        raise ValueError("CUDA device allocations must be unique non-negative integers")
    return devices


def _wait_for_caller(process: subprocess.Popen, base_url: str, *, timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("frozen caller service failed during startup")
        try:
            with urlopen(base_url + "/health", timeout=2) as response:  # noqa: S310 - loopback only
                payload = json.loads(response.read())
            if payload.get("ready") is True and payload.get("frozen") is True:
                return
        except Exception:
            time.sleep(0.5)
    raise TimeoutError("frozen caller service did not become ready before the startup deadline")


def _relay_trainer_process(
    process: subprocess.Popen[str], caller: subprocess.Popen[Any],
    experiment: GPUWorkerExperimentV1, events: EventWriter, *,
    stdout_path: Path, stderr_path: Path,
) -> bool:
    """Persist and relay trainer output without inventing unavailable metrics."""

    if process.stdout is None or process.stderr is None:
        raise RuntimeError("trainer process pipes were not configured")
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ, ("stdout", stdout_path))
    selector.register(process.stderr, selectors.EVENT_READ, ("stderr", stderr_path))
    streams = {
        stdout_path: stdout_path.open("w", encoding="utf-8"),
        stderr_path: stderr_path.open("w", encoding="utf-8"),
    }
    cancelled = False
    try:
        while selector.get_map():
            if experiment.cancellation_file and Path(experiment.cancellation_file).exists():
                cancelled = True
                _stop_process(process)
            if caller.poll() is not None and process.poll() is None:
                _stop_process(process)
                raise RuntimeError("frozen caller service exited during training")
            ready = selector.select(timeout=0.5)
            for key, _ in ready:
                channel, path = key.data
                line = key.fileobj.readline()
                if line:
                    safe_line = _redact_line(line)
                    streams[path].write(safe_line); streams[path].flush()
                    payload = _structured_trainer_line(safe_line)
                    if payload is not None:
                        events.emit("metric", **payload)
                    else:
                        events.emit("log", channel=channel, message=safe_line.rstrip("\n"))
                else:
                    selector.unregister(key.fileobj)
            if process.poll() is not None and not ready:
                # EOF will become readable; keep one bounded pass to drain it.
                for key in list(selector.get_map().values()):
                    channel, path = key.data
                    line = key.fileobj.readline()
                    if line:
                        safe_line = _redact_line(line)
                        streams[path].write(safe_line); streams[path].flush()
                        payload = _structured_trainer_line(safe_line)
                        events.emit("metric", **payload) if payload else events.emit(
                            "log", channel=channel, message=safe_line.rstrip("\n"),
                        )
                    else:
                        selector.unregister(key.fileobj)
        process.wait()
        return cancelled
    finally:
        selector.close()
        for stream in streams.values():
            stream.close()


def _structured_trainer_line(line: str) -> dict[str, Any] | None:
    """Accept only the narrow JSON metric envelope emitted by the GPU plugin."""

    try:
        value = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict) or value.get("type") != "metric":
        return None
    name, number = value.get("name"), value.get("value")
    if not isinstance(name, str) or not isinstance(number, (int, float)):
        return None
    return {
        "step": int(value.get("step", 0)), "name": name, "value": float(number),
        "source": str(value.get("source", "verl-omni")),
    }


def _redact_line(line: str) -> str:
    safe = line
    for name, value in os.environ.items():
        upper = name.upper()
        if value and len(value) >= 8 and any(marker in upper for marker in ("TOKEN", "SECRET", "PASSWORD", "API_KEY")):
            safe = safe.replace(value, f"[REDACTED:{name}]")
    return safe


def _stop_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill(); process.wait()


def _latest_checkpoint(root: Path) -> Path | None:
    candidates = [
        item for item in root.rglob("*")
        if item.name.startswith(("global_step_", "checkpoint_", "checkpoint-"))
    ]
    return max(candidates, key=lambda item: item.stat().st_mtime_ns) if candidates else None


def _hash_path(path: Path) -> str:
    if path.is_file():
        return sha256(path.read_bytes()).hexdigest()
    digest = sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(str(child.relative_to(path)).encode())
        digest.update(b"\0"); digest.update(child.read_bytes())
    return digest.hexdigest()


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
