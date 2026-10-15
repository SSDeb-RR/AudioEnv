from hashlib import sha256

import pytest
from pydantic import ValidationError

from audioenv.contracts import GenerationRecord, TrajectoryStepV1, TrajectoryV1
from audioenv.reference import build_reference_package
from audioenv.runtime import AudioEnvironment
from audioenv.spec import SpeakAction


H = "0" * 64


def generation(**updates):
    values = {
        "generation_id": "g1",
        "model_id": "tiny",
        "model_revision": "r1",
        "policy_version": "p1",
        "semantic_text": "done",
        "input_ids": (1, 2),
        "generated_ids": (3, 4),
        "attention_mask": (1, 1, 1, 1),
        "action_mask": (1, 1),
        "old_logprobs": (-0.2, -0.3),
    }
    values.update(updates)
    return GenerationRecord(**values)


def test_generation_alignment_rejects_mismatched_logprobs():
    with pytest.raises(ValidationError, match="old_logprobs"):
        generation(old_logprobs=(-0.2,))


def test_generation_rejects_non_finite_values():
    with pytest.raises(ValidationError, match="finite"):
        generation(old_logprobs=(-0.2, float("nan")))


def test_trajectory_requires_terminal_or_truncation():
    step = TrajectoryStepV1(
        index=1,
        generation=generation(),
        action={"kind": "speak"},
        observation={},
        state_before_hash=H,
        state_after_hash=H,
        reward_events=(),
        reward=0,
        done=False,
    )
    with pytest.raises(ValidationError, match="terminal step"):
        TrajectoryV1(
            experiment_id="e1",
            environment_hash=H,
            verifier_hash=H,
            task_id="t1",
            episode_id="episode",
            seed=1,
            model_id="tiny",
            checkpoint_id="base",
            policy_version="p1",
            processor_config={},
            audio_artifacts=(),
            steps=(step,),
            terminal_score=0,
        )


def test_v1_runtime_rejects_unbound_spoken_text():
    env = AudioEnvironment(build_reference_package())
    env.reset("retail-refund-001")
    audio = b"agent-audio"
    record = generation(
        semantic_text="real text",
        audio_artifact_id=sha256(audio).hexdigest(),
    )
    with pytest.raises(ValueError, match="semantic text"):
        env.step_generation(SpeakAction(audio, "fabricated reward text", True), record)

