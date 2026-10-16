from __future__ import annotations

from hashlib import sha256
import asyncio
import io
import json
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
from audioenv.harnesses import build_builtin_registry
from audioenv.platform.gpu import LocalCUDAExecutionProvider
from audioenv.platform.runpod import RunPodConfig, RunPodSecureCloudProvider
from audioenv.reference import build_reference_package
from audioenv.rollout_v2 import ModelDrivenAudioRolloutWorker, V2RolloutRequest
from audioenv.models.caller import ScriptedAudioCallerAdapter
from audioenv.spec import SpeakAction, ToolCallAction
from audioenv.training.audioenv_agent_loop import AudioEnvAgentLoop
from audioenv.training.verl_omni import VerlOmniConfig, VerlOmniGRPOTrainer, VerlOmniGSPOTrainer
from audioenv.training.verl_plugin import AudioEnvVerlAgentLoop, compute_audioenv_score, trajectory_to_verl_payload
from audioenv.training.verl_runtime import NativeVerlRollout
from audioenv.training.qwen_tau2_runtime import _flatten_policy_rows
from audioenv.training.verl_trajectory_bridge import _persist_trajectory, finalize_native_trajectory


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


def test_trajectory_v2_preserves_multichannel_audio_rope_positions():
    original = _trajectory(1.0, 1)
    turn = original.turns[0].model_copy(update={
        "position_ids": ((0, 1, 2, 3), (0, 0, 1, 1), (0, 0, 0, 1)),
    })
    trace = original.model_copy(update={"turns": (turn,)})
    peer = trace.model_copy(update={"episode_id": "ep-2", "seed": 2})
    batch = AudioEnvAgentLoop.to_verl_batch((trace, peer))
    assert batch.position_ids[0] == turn.position_ids
    with pytest.raises(ValueError, match="multimodal position channels"):
        TrainingTurnV2(**{
            **turn.model_dump(),
            "position_ids": ((0, 1), (0, 1, 2, 3)),
        })


def test_native_trajectory_is_persisted_per_worker(monkeypatch, tmp_path):
    monkeypatch.setenv("AUDIOENV_TRAJECTORY_DIR", str(tmp_path))
    trajectory = _trajectory(1.0, 9)
    path = _persist_trajectory(trajectory)
    assert path is not None and path.parent == tmp_path
    row = json.loads(path.read_text().splitlines()[0])
    assert row["schema_version"] == "audioenv.trajectory/v2"
    assert row["task_id"] == trajectory.task_id


def test_trainer_registry_keeps_grpo_experimental_until_separate_acceptance():
    registry = build_builtin_registry()
    gspo = registry.get("qwen3-omni-thinker-gspo-lora")
    grpo = registry.get("qwen3-omni-thinker-grpo-lora")
    assert gspo is not None and gspo.release_tier == "release_candidate"
    assert grpo is not None and grpo.release_tier == "experimental"
    assert "Experimental" in grpo.display_name

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
    assert command[:4] == ("uvx", "--from", "repo2rlenv==0.9.3", "repo2rlenv")
    assert kwargs["bootstrap_network"] is True
    assert kwargs["policy"].host_socket is False
    assert kwargs["policy"].inherit_secrets is False
    assert report["commit"] == "e5e6a93a6d7947a012a361b68457a982cc85fc9e"
    with pytest.raises(RuntimeError, match="refuses to mount"):
        Repo2RLEnvGenerator(sandbox).generate(source, output, pipeline="pr_runtime")


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

    # VeRL receives one append-only sequence: model tokens are trainable while
    # caller/tool observation tokens inserted between turns are mask zero.
    turns = list(trace.turns)
    rebuilt = []
    context = tuple(turns[0].input_ids)
    for index, turn in enumerate(turns):
        if index:
            observation = (80 + index,)
            context = context + tuple(turns[index - 1].generated_ids) + observation
            turns[index] = turn.model_copy(update={
                "input_ids": context,
                "attention_mask": (1,) * (len(context) + len(turn.generated_ids)),
                "position_ids": tuple(range(len(context) + len(turn.generated_ids))),
            })
        rebuilt.append(turns[index])
    trace = trace.model_copy(update={"turns": tuple(rebuilt)})
    payload = trajectory_to_verl_payload(trace)
    assert payload.response_mask.count(1) == sum(len(turn.generated_ids) for turn in trace.turns)
    assert payload.response_mask.count(0) == len(trace.turns) - 1
    assert payload.extra_fields["audioenv_trusted_reward"]["verifier_hash"] == trace.verifier_hash


def test_gspo_and_grpo_emit_distinct_real_loss_modes(tmp_path: Path, monkeypatch):
    checkout = tmp_path / "verl"; checkout.mkdir()
    model = tmp_path / "model"; model.mkdir()
    data = tmp_path / "rollouts.parquet"; data.write_bytes(b"fixture")
    validation = tmp_path / "validation.parquet"; validation.write_bytes(b"fixture")
    monkeypatch.setattr("audioenv.training.verl_omni._git_revision", lambda _: "49dedb6ec0691588494b629fc1b610a86612966b")
    base = dict(
        checkout=checkout, model_path=model, dataset_path=data,
        validation_dataset_path=validation, output_dir=tmp_path / "out",
    )
    gspo = VerlOmniGSPOTrainer().launch_command(VerlOmniConfig(**base, algorithm="gspo"), require_preflight=False)
    grpo = VerlOmniGRPOTrainer().launch_command(VerlOmniConfig(**base, algorithm="grpo"), require_preflight=False)
    assert "actor_rollout_ref.actor.policy_loss.loss_mode=gspo" in gspo
    assert "actor_rollout_ref.actor.policy_loss.loss_mode=vanilla" in grpo
    assert "algorithm.adv_estimator=grpo" in gspo and "algorithm.adv_estimator=grpo" in grpo
    assert gspo[:3] == ("python3", "-m", "verl_omni.trainer.main_omni")
    assert not any(item.startswith("--config-") for item in gspo)
    assert "actor_rollout_ref.rollout.name=vllm_omni" in gspo
    assert "+actor_rollout_ref.rollout.engine_kwargs.vllm_omni.pipeline_name=qwen3_omni_moe" in gspo
    assert "+actor_rollout_ref.rollout.engine_kwargs.vllm_omni.pipeline_mode=full" in gspo
    assert "actor_rollout_ref.rollout.agent.default_agent_loop=audioenv_agent" in gspo
    assert "reward.custom_reward_function.path=pkg://audioenv.training.verl_plugin" in gspo
    assert "trainer.val_before_train=true" in gspo
    assert "trainer.test_freq=1" in gspo
    assert "trainer.save_freq=1" in gspo
    checkpoint = base["output_dir"] / "global_step_1"; checkpoint.mkdir(parents=True)
    reload_command = VerlOmniGSPOTrainer().reload_evaluation_command(
        VerlOmniConfig(**base, algorithm="gspo"), checkpoint,
    )
    assert "trainer.val_only=true" in reload_command
    assert f"trainer.resume_from_path={checkpoint.resolve()}" in reload_command
    assert f"data.train_files={data.resolve()}" in gspo
    assert f"data.val_files={validation.resolve()}" in gspo

    same = VerlOmniConfig(
        checkout=checkout, model_path=model, dataset_path=data,
        validation_dataset_path=data, output_dir=tmp_path / "out", algorithm="gspo",
    )
    assert any("different files" in item for item in VerlOmniGSPOTrainer().validate_config(same))


def test_verl_reward_uses_only_hash_bound_protected_payload():
    extra = {
        "audioenv_verifier_hash": "abc",
        "audioenv_trusted_reward": {
            "terminal": 1.0, "process": 0.25, "verifier_hash": "abc",
        },
        "audioenv_rollout_draft": {"task_id": "task-1", "seed": 7, "audio_profile_id": "native-clean"},
    }
    result = compute_audioenv_score("audioenv/tau2-retail", "ignore me", {"fake": "gold"}, extra)
    assert result["score"] == 1.25
    assert result["task_id"] == "task-1" and result["seed"] == 7
    bad = {**extra, "audioenv_verifier_hash": "different"}
    with pytest.raises(ValueError, match="verifier hash mismatch"):
        compute_audioenv_score("audioenv/tau2-retail", "", None, bad)

    trajectory = _trajectory(1.0, 9)
    payload = trajectory_to_verl_payload(trajectory)
    assert payload.reward_score is None
    assert payload.extra_fields["audioenv_verifier_hash"] == trajectory.verifier_hash


def test_verl_native_runtime_refuses_implicit_scripted_fallback(monkeypatch):
    from audioenv.training.verl_runtime import _runtime_factory

    monkeypatch.delenv("AUDIOENV_NATIVE_RUNTIME_FACTORY", raising=False)
    with pytest.raises(RuntimeError, match="reviewed Qwen caller/environment runtime"):
        _runtime_factory()


def test_native_verl_rollout_preserves_audio_for_processor_derived_positions():
    native = NativeVerlRollout(
        prompt_ids=(1, 2), response_ids=(3, 4, 5), response_mask=(1, 0, 1),
        response_logprobs=(-0.1, 0.0, -0.2), reward_score=1.0, num_turns=2,
        audio_paths=("/immutable/caller-a.wav", "/immutable/caller-b.wav"),
        mm_processor_kwargs={"sampling_rate": 16_000},
        extra_fields={"audioenv_trusted_reward": {"terminal": 1.0}},
    )

    async def factory(**_):
        return native

    loop = AudioEnvVerlAgentLoop(rollout_factory=factory)
    output = asyncio.run(loop.run({"temperature": 1.0}))
    assert output["multi_modal_data"] == {"audios": list(native.audio_paths)}
    assert output["mm_processor_kwargs"] == {"sampling_rate": 16_000}
    assert output["response_mask"] == [1, 0, 1]
    assert "position_ids" not in output


def test_qwen_runtime_flattens_only_policy_tokens_into_training_mask():
    prompt, response, mask, logprobs = _flatten_policy_rows((
        ((10, 11), (12,), (-0.1,)),
        ((10, 11, 12, 20, 21), (22, 23), (-0.2, -0.3)),
    ))
    assert prompt == (10, 11)
    assert response == (12, 20, 21, 22, 23)
    assert mask == (1, 0, 0, 1, 1)
    assert logprobs == (-0.1, 0.0, 0.0, -0.2, -0.3)

    with pytest.raises(RuntimeError, match="append-only"):
        _flatten_policy_rows((((1,), (2,), (-0.1,)), ((9,), (3,), (-0.2,))))


def test_verl_postprocessing_finalizes_real_multichannel_positions(tmp_path: Path):
    import torch

    audio_path = tmp_path / "caller.wav"
    audio_path.write_bytes(_wav())
    draft = {
        "schema_version": "audioenv.native-rollout-draft/v1",
        "experiment_id": "exp", "environment_hash": "a" * 64,
        "verifier_hash": "b" * 64, "reward_version": "tau2-v1",
        "task_id": "0", "episode_id": "episode", "seed": 7,
        "model_id": "qwen", "checkpoint_id": "base", "policy_version": "p1",
        "terminal_reward": 1.0, "process_reward": 0.0,
        "truncated": False, "truncation_reason": None,
        "audio_paths": [str(audio_path)], "actions": [{"kind": "tool"}],
    }
    # Two left-padding positions, a two-token prompt, then three response slots.
    padded = torch.tensor([[
        [90, 91, 0, 1, 2, 3, 4],
        [90, 91, 10, 11, 12, 13, 14],
        [90, 91, 20, 21, 22, 23, 24],
        [90, 91, 30, 31, 32, 33, 34],
    ]])
    trace = finalize_native_trajectory(
        draft=draft, prompt_ids=(99, 99, 2), response_ids=(3, 4, 5),
        response_mask=(1, 0, 1), response_logprobs=(-0.1, 0.0, -0.2),
        padded_position_ids=padded, prompt_width=4, audio_token_id=99,
    )
    assert trace.terminal_reward == 1 and trace.done
    assert trace.turns[0].position_ids == (
        (91, 0, 1, 2, 3, 4), (91, 10, 11, 12, 13, 14),
        (91, 20, 21, 22, 23, 24), (91, 30, 31, 32, 33, 34),
    )
    assert trace.turns[0].response_mask == (1, 0, 1)
    assert trace.turns[0].multimodal_features[0].start_token == 0
    assert trace.turns[0].multimodal_features[0].end_token == 2


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
    config = RunPodConfig(
        run_name="paid-proof", image_digest="sha256:" + "a" * 64,
        max_runtime_hours=2, max_total_budget_usd=40,
    )
    quote = provider.quote(config)
    assert quote.maximum_cost_usd == 32
    with pytest.raises(ValueError, match="exactly match"):
        provider.provision(config, quote, confirmation_name="wrong")
    assert transport.created == []
    provider.provision(config, quote, confirmation_name="paid-proof")
    assert transport.created[0]["cloud"] == "SECURE"
    assert transport.created[0]["gpu"] == {"id": "H100", "count": 8, "minRamPerGpu": 80}
