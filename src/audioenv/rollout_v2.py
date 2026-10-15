"""Model-driven caller/agent collection for training-grade trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import io
import math
import wave

from .adapter import EnvironmentPackage
from .artifacts import LocalArtifactStore
from .contracts import (
    AudioArtifactV1, MultimodalFeatureRefV1, TrainingTurnV2, TrajectoryV2,
)
from .evaluation import package_hashes
from .models.base import V1ModelAdapter
from .models.caller import CallerAdapter
from .runtime import AudioEnvironment
from .spec import AudioObservation, InvalidAction, SpeakAction
from .trace import RewardEvent


@dataclass(frozen=True, slots=True)
class V2RolloutRequest:
    experiment_id: str
    task_id: str
    seed: int
    checkpoint_id: str
    policy_version: str
    reward_version: str
    processor_config: dict
    audio_profile_id: str | None = None
    max_turns: int = 32


class ModelDrivenAudioRolloutWorker:
    def __init__(
        self, package: EnvironmentPackage, agent: V1ModelAdapter,
        caller: CallerAdapter, artifacts: LocalArtifactStore,
    ):
        self.package, self.agent, self.caller, self.artifacts = package, agent, caller, artifacts

    def run(self, request: V2RolloutRequest, *, cancel_check=None) -> TrajectoryV2:
        agent_probe, caller_probe = self.agent.probe(), self.caller.probe()
        reasons = tuple(agent_probe.reasons if not agent_probe.available else ()) + tuple(
            caller_probe.reasons if not caller_probe.available else ()
        )
        if reasons:
            raise RuntimeError("model-driven rollout preflight failed: " + "; ".join(reasons))
        initial = self.caller.reset(request.seed)
        runtime = AudioEnvironment(self.package, external_caller=True)
        observation = runtime.reset(
            request.task_id, request.seed, audio_profile_id=request.audio_profile_id,
            model_id=self.agent.model_id, initial_audio=initial.audio,
        )
        audio = [self._store_audio(initial.audio, "caller")]
        turns: list[TrainingTurnV2] = []
        process_reward = 0.0
        truncation = None
        for index in range(1, request.max_turns + 1):
            if cancel_check and cancel_check():
                truncation = "cancelled"
                break
            action, generation = self.agent.generate_step(observation)
            if generation.policy_version != request.policy_version:
                raise ValueError("policy version drift")
            if generation.semantic_text.lstrip().startswith("{") and generation.tool_call is None:
                action = InvalidAction(
                    sha256(generation.semantic_text.encode()).hexdigest(), "malformed_structured_envelope",
                )
            agent_audio = None
            if isinstance(action, SpeakAction) and action.audio:
                artifact = self._store_audio(action.audio, "agent")
                if generation.audio_artifact_id != artifact.artifact_id:
                    raise ValueError("generated speech is not bound to its audio artifact")
                audio.append(artifact)
                agent_audio = action.audio
            observation = runtime.step_generation(action, generation)
            events = runtime.trace.steps[-1].reward_events
            process_reward += sum(item.value for item in events if item.category != "terminal")
            input_audio = audio[-2] if agent_audio and len(audio) >= 2 else audio[-1]
            positions = generation.position_ids or tuple(range(len(generation.attention_mask)))
            feature_end = min(len(generation.input_ids), len(generation.attention_mask))
            turns.append(TrainingTurnV2(
                index=index, input_ids=generation.input_ids, generated_ids=generation.generated_ids,
                attention_mask=generation.attention_mask, response_mask=generation.action_mask,
                position_ids=positions, rollout_logprobs=generation.old_logprobs,
                multimodal_features=(MultimodalFeatureRefV1(
                    kind="audio", artifact_id=input_audio.artifact_id,
                    start_token=0, end_token=max(feature_end, 1),
                ),), action=runtime.trace.steps[-1].action,
                reward_events=tuple(_reward(item) for item in events), done=observation.done,
            ))
            if observation.done:
                break
            if agent_audio is not None:
                caller_turn = self.caller.respond(agent_audio)
                if caller_turn is None:
                    truncation = "caller_ended"
                    break
                caller_artifact = self._store_audio(caller_turn.audio, "caller")
                audio.append(caller_artifact)
                observation = AudioObservation(
                    audio=caller_turn.audio, reward=observation.reward,
                    done=False, metrics=observation.metrics,
                )
        else:
            truncation = "max_turns"
        environment_hash, verifier_hash = package_hashes(self.package)
        terminal = float(bool(runtime.trace.final_metrics.get("terminal_pass"))) if runtime.trace.final_metrics else 0.0
        return TrajectoryV2(
            experiment_id=request.experiment_id, environment_hash=environment_hash,
            verifier_hash=verifier_hash, reward_version=request.reward_version,
            task_id=request.task_id, episode_id=runtime.trace.episode_id, seed=request.seed,
            model_id=self.agent.model_id, checkpoint_id=request.checkpoint_id,
            policy_version=request.policy_version, processor_config=request.processor_config,
            audio_artifacts=tuple(_deduplicate(audio)), turns=tuple(turns),
            terminal_reward=terminal, process_reward=process_reward,
            done=bool(runtime.trace.final_metrics), truncation_reason=truncation,
        )

    def _store_audio(self, payload: bytes, producer: str) -> AudioArtifactV1:
        stored = self.artifacts.put_bytes(payload, media_type="audio/wav")
        with wave.open(io.BytesIO(payload), "rb") as wav:
            rate, frames = wav.getframerate(), wav.getnframes()
        duration = frames * 1000.0 / rate
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("invalid audio duration")
        return AudioArtifactV1(
            artifact_id=stored.id, sha256=stored.sha256, media_type="audio/wav",
            sample_rate_hz=rate, duration_ms=duration, producer=producer,
        )


def _reward(item: RewardEvent) -> dict:
    return {"name": item.name, "value": item.value, "category": item.category, "details": item.details}


def _deduplicate(items):
    seen = set()
    for item in items:
        if item.artifact_id not in seen:
            seen.add(item.artifact_id)
            yield item
