from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from audioenv.contracts import GPUWorkerExperimentV1
from audioenv.platform.gpu_worker import (
    EventWriter, QwenGPUAllocation, _hash_path, _latest_checkpoint,
    _redact_line, _relay_trainer_process, _structured_trainer_line, load_experiment, run_experiment,
)
from audioenv.platform.gpu_runtime import attest_qwen_runtime
from audioenv.training.tau2_dataset import build_tau2_rollout_dataset


def test_worker_bundle_rejects_incompatible_backend():
    with pytest.raises(ValueError, match="local REINFORCE"):
        GPUWorkerExperimentV1(
            experiment_id="bad", run_name="bad", backend="tiny-cpu-proof",
            trainer="qwen3-omni-thinker-gspo-lora", algorithm="gspo", output_dir="out",
        )


def test_provider_neutral_worker_runs_training_and_emits_artifacts(tmp_path: Path):
    bundle = GPUWorkerExperimentV1(
        experiment_id="worker-proof", run_name="worker-proof", backend="tiny-cpu-proof",
        trainer="audioenv/small-audio-reinforce", algorithm="reinforce",
        output_dir=str(tmp_path / "output"),
        hyperparameters={"steps": 2, "checkpoint_interval": 1, "seed": 9},
        dependency_pins={"audioenv": "test"},
    )
    path = tmp_path / "experiment.json"
    path.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")
    assert load_experiment(path).content_hash == bundle.content_hash
    assert run_experiment(bundle) == 0
    lines = [json.loads(line) for line in (tmp_path / "output" / "events.jsonl").read_text().splitlines()]
    event_types = [line["type"] for line in lines]
    assert event_types[0] == "worker.started"
    assert "preflight.completed" in event_types
    assert "training.started" in event_types
    assert "checkpoint.saved" in event_types
    assert "evaluation.completed" in event_types
    assert event_types[-1] == "worker.completed"
    assert (tmp_path / "output" / "experiment.snapshot.json").is_file()
    manifest = json.loads((tmp_path / "output" / "worker-artifacts.json").read_text())
    assert any(row["path"] == "final.pt" for row in manifest)


def test_worker_cancellation_is_a_terminal_event(tmp_path: Path):
    cancel = tmp_path / "cancel"
    cancel.write_text("stop", encoding="utf-8")
    bundle = GPUWorkerExperimentV1(
        experiment_id="cancel-proof", run_name="cancel-proof", backend="tiny-cpu-proof",
        trainer="audioenv/small-audio-reinforce", algorithm="reinforce",
        output_dir=str(tmp_path / "cancelled"), cancellation_file=str(cancel),
        hyperparameters={"steps": 5, "checkpoint_interval": 1},
    )
    assert run_experiment(bundle) == 130
    events = (tmp_path / "cancelled" / "events.jsonl").read_text()
    assert '"type":"worker.cancelled"' in events


def test_qwen_gpu_allocation_is_explicit_and_disjoint():
    allocation = QwenGPUAllocation.from_hyperparameters({})
    assert allocation.trainer == "0,1,2,3"
    assert allocation.caller == "4,5"
    with pytest.raises(ValueError, match="disjoint"):
        QwenGPUAllocation.from_hyperparameters({
            "trainer_cuda_devices": "0,1,2,3", "caller_cuda_devices": "3,4",
        })


class _Discovery:
    def discover(self):
        return {"tasks": [
            {"id": "train-1", "split": "train"},
            {"id": "test-1", "split": "test"},
        ]}


def _qwen_attestation_fixture(tmp_path: Path):
    lock = json.loads((Path(__file__).parents[1] / "docker/gpu-worker/dependencies.lock.json").read_text())
    lock_path = tmp_path / "lock.json"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    model = tmp_path / "model"; model.mkdir()
    (model / "audioenv-model-revision.json").write_text(
        json.dumps({"revision": lock["qwen_model_revision"]}), encoding="utf-8",
    )
    train = tmp_path / "train.jsonl"; test = tmp_path / "test.jsonl"
    common = dict(
        client=_Discovery(), experiment_id="exp", environment_hash="a" * 64,
        verifier_hash="b" * 64, policy_version="p1", checkpoint_id="base",
    )
    build_tau2_rollout_dataset(output=train, split="train", **common)
    build_tau2_rollout_dataset(output=test, split="test", **common)
    manifest = tmp_path / "environment.json"; manifest.write_text("{}", encoding="utf-8")
    checkout = tmp_path / "checkout"; checkout.mkdir()
    bundle = GPUWorkerExperimentV1(
        experiment_id="exp", run_name="gpu-proof", backend="qwen3-omni-verl",
        trainer="qwen3-omni-thinker-gspo-lora", algorithm="gspo",
        environment_manifest=str(manifest), model_path=str(model), caller_model_path=str(model),
        rollout_dataset_path=str(train), validation_dataset_path=str(test),
        verl_checkout=str(checkout), tau2_checkout=str(checkout), output_dir=str(tmp_path / "out"),
        dependency_pins={key: str(value) for key, value in lock.items() if key not in {"schema_version", "base_image_policy"}},
    )
    return bundle, lock_path


def test_qwen_runtime_attests_lock_model_and_disjoint_datasets(tmp_path: Path, monkeypatch):
    bundle, lock_path = _qwen_attestation_fixture(tmp_path)
    monkeypatch.setattr("audioenv.platform.gpu_runtime._git_revision", lambda _path: bundle.dependency_pins["tau2_commit"])
    report = attest_qwen_runtime(bundle, lock_path=lock_path, probe_cuda=False, probe_packages=False)
    assert report.available, report.reasons
    assert report.evidence["rollout_dataset"]["split"] == "train"
    assert report.evidence["validation_dataset"]["split"] == "test"


def test_qwen_runtime_rejects_dependency_and_dataset_tampering(tmp_path: Path, monkeypatch):
    bundle, lock_path = _qwen_attestation_fixture(tmp_path)
    monkeypatch.setattr("audioenv.platform.gpu_runtime._git_revision", lambda _path: bundle.dependency_pins["tau2_commit"])
    altered = bundle.model_copy(update={"dependency_pins": {**bundle.dependency_pins, "vllm_omni_commit": "wrong"}})
    Path(bundle.rollout_dataset_path).write_text("{}\n", encoding="utf-8")
    report = attest_qwen_runtime(altered, lock_path=lock_path, probe_cuda=False, probe_packages=False)
    assert not report.available
    assert any("vllm_omni_commit" in reason for reason in report.reasons)
    assert any("dataset hash" in reason for reason in report.reasons)


def test_gpu_image_builder_rejects_mutable_base_before_docker(tmp_path: Path):
    wheel = tmp_path / "audioenv.whl"; wheel.write_bytes(b"wheel")
    script = Path(__file__).parents[1] / "docker/gpu-worker/build.sh"
    result = subprocess.run(
        [str(script), "nvidia/cuda:latest", str(wheel), "audioenv:test"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 2
    assert "immutable sha256" in result.stderr


def test_gpu_image_copies_dependency_receipt_and_installs_pinned_sources():
    dockerfile = (Path(__file__).parents[1] / "docker/gpu-worker/Dockerfile").read_text()
    assert "dependencies.lock.json /opt/audioenv/dependencies.lock.json" in dockerfile
    assert "vllm-project/vllm-omni.git@${VLLM_OMNI_COMMIT}" in dockerfile
    assert "verl-project/verl.git@${VERL_COMMIT}" in dockerfile


class _LivingCaller:
    def poll(self): return None


def test_trainer_output_relay_persists_logs_and_emits_real_metrics(tmp_path: Path):
    bundle = GPUWorkerExperimentV1(
        experiment_id="relay", run_name="relay", backend="tiny-cpu-proof",
        trainer="audioenv/small-audio-reinforce", algorithm="reinforce",
        output_dir=str(tmp_path),
    )
    script = (
        "import sys; "
        "print('plain output', flush=True); "
        "print('{\"type\":\"metric\",\"step\":2,\"name\":\"loss\",\"value\":0.25}', flush=True); "
        "print('warning', file=sys.stderr, flush=True)"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", script], stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, bufsize=1,
    )
    event_path = tmp_path / "events.jsonl"
    cancelled = _relay_trainer_process(
        process, _LivingCaller(), bundle, EventWriter("relay", event_path),
        stdout_path=tmp_path / "stdout.log", stderr_path=tmp_path / "stderr.log",
    )
    assert not cancelled and process.returncode == 0
    events = [json.loads(line) for line in event_path.read_text().splitlines()]
    assert any(row["type"] == "metric" and row["payload"]["name"] == "loss" for row in events)
    assert any(row["type"] == "log" and row["payload"]["channel"] == "stderr" for row in events)
    assert "plain output" in (tmp_path / "stdout.log").read_text()
    assert _structured_trainer_line("not json") is None


def test_checkpoint_discovery_and_tree_hash_are_deterministic(tmp_path: Path):
    checkpoint = tmp_path / "global_step_1"; checkpoint.mkdir()
    (checkpoint / "adapter.bin").write_bytes(b"weights")
    assert _latest_checkpoint(tmp_path) == checkpoint
    assert _hash_path(checkpoint) == _hash_path(checkpoint)


def test_streamed_logs_redact_secret_environment_values(monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "super-secret-value")
    assert _redact_line("token=super-secret-value") == "token=[REDACTED:RUNPOD_API_KEY]"
