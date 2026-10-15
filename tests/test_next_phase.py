from __future__ import annotations

from hashlib import sha256
import io
from pathlib import Path
import wave

import pytest

from audioenv.adapters.harbor import HarborEnvironmentAdapter
from audioenv.artifacts import LocalArtifactStore
from audioenv.contracts import (
    AudioArtifactV1,
    GenerationRecord,
    GPUExecutionPlanV1,
    ModelCapabilities,
    MultimodalFeatureRefV1,
    ProbeResult,
    PromotionReportV1,
    TrainingTurnV2,
    TrajectoryV2,
)
from audioenv.integrations.repo2rlenv import Repo2RLEnvGenerator, SandboxResult
from audioenv.platform.gpu import LocalCUDAExecutionProvider
from audioenv.platform.runpod import RunPodConfig, RunPodSecureCloudProvider
from audioenv.reference import build_reference_package
from audioenv.rollout_v2 import ModelDrivenAudioRolloutWorker, V2RolloutRequest
from audioenv.models.caller import ScriptedAudioCallerAdapter
from audioenv.spec import SpeakAction, ToolCallAction
from audioenv.training.audioenv_agent_loop import AudioEnvAgentLoop
from audioenv.training.verl_omni import VerlOmniConfig, VerlOmniGRPOTrainer, VerlOmniGSPOTrainer


HEX = "a" * 64


def _wav() -> bytes:
    stream = io.BytesIO()
    with wave.open(stream, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(b"\0\0" * 160)
    return stream.getvalue()


def _trajectory(reward: float, seed: int) -> TrajectoryV2:
    payload = _wav()
    digest = sha256(payload).hexdigest()
    audio = AudioArtifactV1(
        artifact_id=digest, sha256=digest, media_type="audio/wav",
        sample_rate_hz=16_000, duration_ms=10, producer="caller",
    )
    turn = TrainingTurnV2(
        index=1, input_ids=(1, 2), generated_ids=(3, 4),
        attention_mask=(1, 1, 1, 1), response_mask=(1, 1),
        position_ids=(0, 1, 2, 3), rollout_logprobs=(-0.1, -0.2),
        multimodal_features=(MultimodalFeatureRefV1(
            kind="audio", artifact_id=digest, start_token=0, end_token=2,
        ),), action={"kind": "tool_call"}, reward_events=(), done=True,
    )
    return TrajectoryV2(
        experiment_id="exp", environment_hash=HEX, verifier_hash="b" * 64,
        reward_version="v1", task_id="task", episode_id=f"ep-{seed}", seed=seed,
        model_id="model", checkpoint_id="base", policy_version="p1",
        processor_config={}, audio_artifacts=(audio,), turns=(turn,),
        terminal_reward=reward, process_reward=0, done=True,
    )


def test_promotion_requires_all_gates_and_immutable_verifier():
    checks = {name: True for name in (
        "sandbox_build", "oracle_pass", "noop_fail", "adversarial_fail",
        "determinism", "state_isolation", "leakage", "reward_integrity",
    )}
    report = PromotionReportV1.evaluate(
        environment_hash=HEX, verifier_hash_before="b" * 64,
        verifier_hash_after="b" * 64, checks=checks,
    )
    assert report.promoted
    changed = PromotionReportV1.evaluate(
        environment_hash=HEX, verifier_hash_before="b" * 64,
        verifier_hash_after="c" * 64, checks=checks,
    )
    assert not changed.promoted
    assert changed.missing_or_failed == ("verifier_immutability",)


def test_harbor_import_separates_public_task_and_private_verifier(tmp_path: Path):
    root = tmp_path / "harbor" / "fix-one"
    (root / "environment").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "solution").mkdir()
    (root / "task.toml").write_text('id = "fix-one"\nsplit = "test"\n', encoding="utf-8")
    (root / "instruction.md").write_text("Fix the public bug.", encoding="utf-8")
    (root / "environment" / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
    (root / "tests" / "test.sh").write_text("#!/bin/sh\n# SECRET_EXPECTATION\nexit 0\n", encoding="utf-8")
    (root / "solution" / "solve.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    store = LocalArtifactStore(tmp_path / "artifacts")
    package, bundles = HarborEnvironmentAdapter(store).convert(
        root.parent, environment_id="repo-fixture", source_uri="fixture://repo",
    )
    public = package.model_dump_json()
    assert "SECRET_EXPECTATION" not in public
    protected = store.get_bytes(package.protected_verifier.artifact_id).decode()
    assert "tests/test.sh" in protected
    assert str(tmp_path) not in protected
    assert bundles[0].task_id == "fix-one"


class _FakeSandbox:
    def __init__(self): self.call = None
    def run(self, image, command, **kwargs):
        self.call = (image, command, kwargs)
        return SandboxResult(0, "generated", "")


def test_repo2rlenv_uses_pinned_isolated_protocol(tmp_path: Path):
    source = tmp_path / "repo"; source.mkdir()
    output = tmp_path / "out"
    sandbox = _FakeSandbox()
    report = Repo2RLEnvGenerator(sandbox).generate(source, output, limit=2)
    _, command, kwargs = sandbox.call
    assert command[:2] == ("repo2rlenv", "generate")
    assert kwargs["bootstrap_network"] is True
    assert kwargs["policy"].host_socket is False
    assert kwargs["policy"].inherit_secrets is False
    assert report["commit"] == "e5e6a93a6d7947a012a361b68457a982cc85fc9e"


def test_v2_batch_checks_groups_versions_and_logprob_parity():
    trajectories = (_trajectory(0.0, 1), _trajectory(1.0, 2))
    batch = AudioEnvAgentLoop.to_verl_batch(
        trajectories, recompute_logprobs=lambda _prompt, _generated: (-0.1, -0.2),
    )
    assert batch.response_masks == ((1, 1), (1, 1))
    assert AudioEnvAgentLoop.zero_std_ratio(trajectories) == 0
    with pytest.raises(ValueError, match="log-probability mismatch"):
        AudioEnvAgentLoop.to_verl_batch(
            trajectories, recompute_logprobs=lambda _prompt, _generated: (-4.0, -4.0),
        )
    with pytest.raises(ValueError, match="at least two"):
        AudioEnvAgentLoop.to_verl_batch((trajectories[0],))


class _FakeAudioAgent:
    model_id = "fixture/omni-agent"
    def __init__(self): self.index = 0
    @property
    def capabilities(self):
        return ModelCapabilities(
            audio_input=True, audio_output=True, structured_tools=True,
            token_logprobs=True, trainable=True,
        )
    def probe(self):
        return ProbeResult(
            available=True, model_id=self.model_id, capabilities=self.capabilities,
            device="cpu", revision="fixture",
        )
    def generate_step(self, observation):
        del observation
        actions = (
            ToolCallAction("lookup_order", {"order_id": "A-1042"}),
            SpeakAction(_wav(), "Let me complete that refund.", False),
            ToolCallAction("refund_order", {"order_id": "A-1042", "customer_zip": "94107"}),
            SpeakAction(_wav(), "Refund RF-1001 for $79 is complete.", True),
        )
        action = actions[self.index]; self.index += 1
        tool = None
        audio_id = None
        if isinstance(action, ToolCallAction):
            semantic = '{"tool_call":true}'
            tool = {"tool_call": {"name": action.name, "arguments": action.arguments}}
        else:
            semantic = action.semantic_text
            audio_id = sha256(action.audio).hexdigest()
        record = GenerationRecord(
            generation_id=f"g-{self.index}", model_id=self.model_id,
            model_revision="fixture", policy_version="p1", semantic_text=semantic,
            input_ids=(1, 2), generated_ids=(3,), attention_mask=(1, 1, 1),
            action_mask=(1,), old_logprobs=(-0.1,), position_ids=(0, 1, 2),
            audio_artifact_id=audio_id, tool_call=tool,
        )
        return action, record
    def save_adapter(self, destination): return destination
    def load_adapter(self, source): return None


def test_model_driven_caller_agent_loop_produces_terminal_v2_trace(tmp_path: Path):
    worker = ModelDrivenAudioRolloutWorker(
        build_reference_package(), _FakeAudioAgent(),
        ScriptedAudioCallerAdapter(("Please refund my order.", "Yes, continue.")),
        LocalArtifactStore(tmp_path / "artifacts"),
    )
    trace = worker.run(V2RolloutRequest(
        experiment_id="exp", task_id="retail-refund-001", seed=7,
        checkpoint_id="base", policy_version="p1", reward_version="reference-v2",
        processor_config={"fixture": True},
    ))
    assert trace.done and trace.terminal_reward == 1
    assert len(trace.turns) == 4
    assert {item.producer for item in trace.audio_artifacts} == {"caller", "agent"}
    assert all(turn.response_mask == (1,) for turn in trace.turns)


def test_gspo_and_grpo_emit_distinct_real_loss_modes(tmp_path: Path, monkeypatch):
    checkout = tmp_path / "verl"; checkout.mkdir()
    model = tmp_path / "model"; model.mkdir()
    data = tmp_path / "rollouts.parquet"; data.write_bytes(b"fixture")
    monkeypatch.setattr("audioenv.training.verl_omni._git_revision", lambda _: "49dedb6ec0691588494b629fc1b610a86612966b")
    base = dict(checkout=checkout, model_path=model, dataset_path=data, output_dir=tmp_path / "out")
    gspo = VerlOmniGSPOTrainer().launch_command(VerlOmniConfig(**base, algorithm="gspo"), require_preflight=False)
    grpo = VerlOmniGRPOTrainer().launch_command(VerlOmniConfig(**base, algorithm="grpo"), require_preflight=False)
    assert "actor_rollout_ref.actor.policy_loss.loss_mode=gspo" in gspo
    assert "actor_rollout_ref.actor.policy_loss.loss_mode=vanilla" in grpo
    assert "algorithm.adv_estimator=grpo" in gspo and "algorithm.adv_estimator=grpo" in grpo


class _RunPodTransport:
    def __init__(self): self.created = []; self.stopped = []
    def list_gpu_types(self):
        return [{"id": "H100", "memory_gb": 80, "secure_cloud_available": 8, "price_per_gpu_hour": 2.0}]
    def create_pod(self, payload): self.created.append(payload); return {"id": "pod-1"}
    def get_pod(self, pod_id): return {"id": pod_id, "status": "RUNNING"}
    def stop_pod(self, pod_id): self.stopped.append(pod_id); return {"id": pod_id, "status": "STOPPED"}


def test_gpu_plans_are_guarded_and_never_provision_without_exact_confirmation(tmp_path: Path):
    digest = "sha256:" + "d" * 64
    plan = GPUExecutionPlanV1(
        provider="local-cuda", image="audioenv/gpu:v1", image_digest=digest,
        gpu_type="H100", gpu_count=8, memory_gb_per_gpu=80, storage_gb=500,
        max_runtime_hours=2, max_total_budget_usd=200,
    )
    command = LocalCUDAExecutionProvider.docker_command(plan, data_dir=tmp_path, command=("train",))
    assert "--read-only" in command and "--cap-drop=ALL" in command
    transport = _RunPodTransport(); provider = RunPodSecureCloudProvider(transport)
    config = RunPodConfig(run_name="paid-proof", max_runtime_hours=2, max_total_budget_usd=40)
    quote = provider.quote(config)
    assert quote.maximum_cost_usd == 32
    with pytest.raises(ValueError, match="exactly match"):
        provider.provision(config, quote, confirmation_name="wrong")
    assert transport.created == []
    provider.provision(config, quote, confirmation_name="paid-proof")
    assert transport.created[0]["cloud"] == "SECURE"
    assert transport.created[0]["gpu"] == {"id": "H100", "count": 8, "minRamPerGpu": 80}
