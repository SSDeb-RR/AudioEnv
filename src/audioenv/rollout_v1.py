"""Validated V1 rollout collection with content-addressed audio artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import io
from pathlib import Path
import wave

from .adapter import EnvironmentPackage
from .artifacts import LocalArtifactStore
from .contracts import AudioArtifactV1, TrajectoryStepV1, TrajectoryV1
from .evaluation import package_hashes
from .models.base import V1ModelAdapter
from .runtime import AudioEnvironment
from .spec import SpeakAction
from .trace import state_hash


@dataclass(frozen=True, slots=True)
class V1RolloutRequest:
    experiment_id: str
    task_id: str
    seed: int
    checkpoint_id: str
    policy_version: str
    processor_config: dict
    audio_profile_id: str | None = None


class V1RolloutWorker:
    def __init__(
        self,
        package: EnvironmentPackage,
        model: V1ModelAdapter,
        artifacts: LocalArtifactStore,
    ):
        self.package = package
        self.model = model
        self.artifacts = artifacts

    def run(self, request: V1RolloutRequest, *, cancel_check=None) -> TrajectoryV1:
        probe = self.model.probe()
        if not probe.available:
            raise RuntimeError("model preflight failed: " + "; ".join(probe.reasons))
        environment_hash, verifier_hash = package_hashes(self.package)
        runtime = AudioEnvironment(self.package)
        observation = runtime.reset(
            request.task_id,
            request.seed,
            audio_profile_id=request.audio_profile_id,
            model_id=self.model.model_id,
        )
        audio_records: list[AudioArtifactV1] = []
        if observation.audio:
            audio_records.append(self._store_audio(observation.audio, "caller"))
        steps: list[TrajectoryStepV1] = []
        truncation = None
        while not observation.done:
            if cancel_check and cancel_check():
                truncation = "cancelled"
                break
            before = state_hash(runtime.state_snapshot)
            action, generation = self.model.generate_step(observation)
            if generation.policy_version != request.policy_version:
                raise ValueError("model generation policy version does not match rollout request")
            if isinstance(action, SpeakAction) and action.audio:
                artifact = self._store_audio(action.audio, "agent")
                if generation.audio_artifact_id != artifact.artifact_id:
                    raise ValueError("model generation references a different audio artifact")
                audio_records.append(artifact)
            observation = runtime.step_generation(action, generation)
            if observation.audio:
                audio_records.append(self._store_audio(observation.audio, "caller"))
            trace_step = runtime.trace.steps[-1]
            steps.append(
                TrajectoryStepV1(
                    index=len(steps) + 1,
                    generation=generation,
                    action=trace_step.action,
                    observation=trace_step.observation,
                    state_before_hash=before,
                    state_after_hash=trace_step.state_hash,
                    reward_events=tuple(
                        {
                            "name": event.name,
                            "value": event.value,
                            "category": event.category,
                            "details": event.details,
                        }
                        for event in trace_step.reward_events
                    ),
                    reward=observation.reward,
                    done=observation.done,
                )
            )
        terminal_score = float(bool(runtime.trace.final_metrics.get("terminal_pass"))) if runtime.trace.final_metrics else 0.0
        return TrajectoryV1(
            experiment_id=request.experiment_id,
            environment_hash=environment_hash,
            verifier_hash=verifier_hash,
            task_id=request.task_id,
            episode_id=runtime.trace.episode_id,
            seed=request.seed,
            model_id=self.model.model_id,
            checkpoint_id=request.checkpoint_id,
            policy_version=request.policy_version,
            processor_config=request.processor_config,
            audio_artifacts=tuple(_deduplicate(audio_records)),
            steps=tuple(steps),
            terminal_score=terminal_score,
            truncation_reason=truncation,
        )

    def _store_audio(self, payload: bytes, producer: str) -> AudioArtifactV1:
        stored = self.artifacts.put_bytes(payload, media_type="audio/wav")
        with wave.open(io.BytesIO(payload), "rb") as wav:
            sample_rate = wav.getframerate()
            duration_ms = wav.getnframes() * 1000.0 / sample_rate
        return AudioArtifactV1(
            artifact_id=stored.id,
            sha256=stored.sha256,
            media_type="audio/wav",
            sample_rate_hz=sample_rate,
            duration_ms=duration_ms,
            producer=producer,
        )


def _deduplicate(items: list[AudioArtifactV1]):
    seen = set()
    for item in items:
        if item.artifact_id not in seen:
            seen.add(item.artifact_id)
            yield item
