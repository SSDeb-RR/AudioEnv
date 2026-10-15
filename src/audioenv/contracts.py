"""Versioned V1 boundary contracts.

The existing dataclasses remain the small in-process runtime representation.
These Pydantic models are used at storage, API, trainer, and plugin boundaries.
"""

from __future__ import annotations

from enum import StrEnum
from hashlib import sha256
import json
import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AuthoringStatus(StrEnum):
    MANUAL = "manual"
    GENERATED_NEEDS_REVIEW = "generated_needs_review"
    VALIDATED = "validated"
    UNSUPPORTED = "unsupported"


class EnvironmentType(StrEnum):
    CONVERSATIONAL = "conversational"
    CODING = "coding"


class SourceProvenance(FrozenModel):
    kind: Literal["builtin", "benchmark", "custom", "repository"]
    uri: str | None = None
    revision: str | None = None
    license: str | None = None


class PublicTaskV1(FrozenModel):
    id: str
    split: Literal["train", "validation", "test"]
    goal: str
    tool_names: tuple[str, ...] = ()
    audio_profile_ids: tuple[str, ...] = ()


class ProtectedVerifierRef(FrozenModel):
    artifact_id: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class EnvironmentPackageV1(FrozenModel):
    schema_version: Literal["audioenv.environment/v1"] = "audioenv.environment/v1"
    id: str
    version: str
    description: str
    environment_type: EnvironmentType = EnvironmentType.CONVERSATIONAL
    source: SourceProvenance
    required_capabilities: tuple[str, ...] = ()
    tasks: tuple[PublicTaskV1, ...]
    protected_verifier: ProtectedVerifierRef
    authoring_status: AuthoringStatus
    parent_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    def canonical_payload(self) -> dict[str, Any]:
        data = self.model_dump(mode="json")
        data.pop("content_hash", None)
        return data

    @staticmethod
    def hash_payload(payload: dict[str, Any]) -> str:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return sha256(encoded).hexdigest()

    @model_validator(mode="after")
    def verify_content_hash(self) -> "EnvironmentPackageV1":
        if self.hash_payload(self.canonical_payload()) != self.content_hash:
            raise ValueError("content_hash does not match the canonical package payload")
        task_ids = [task.id for task in self.tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("task ids must be unique")
        return self


class AudioArtifactV1(FrozenModel):
    artifact_id: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    media_type: Literal["audio/wav", "audio/flac", "audio/ogg"]
    sample_rate_hz: int = Field(gt=0)
    duration_ms: float = Field(gt=0)
    producer: Literal["caller", "agent", "fixture", "perturbation"]
    parent_artifact_id: str | None = None
    perturbations: tuple[str, ...] = ()


class ModelCapabilities(FrozenModel):
    audio_input: bool
    audio_output: bool
    structured_tools: bool
    token_logprobs: bool
    trainable: bool
    batching: bool = False
    distributed: bool = False
    trainable_modules: tuple[str, ...] = ()
    checkpoint_format: str | None = None


class ProbeResult(FrozenModel):
    available: bool
    model_id: str
    capabilities: ModelCapabilities
    device: str | None = None
    revision: str | None = None
    reasons: tuple[str, ...] = ()


class GenerationRecord(FrozenModel):
    generation_id: str
    model_id: str
    model_revision: str
    policy_version: str
    semantic_text: str
    input_ids: tuple[int, ...]
    generated_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    action_mask: tuple[int, ...]
    old_logprobs: tuple[float, ...]
    position_ids: tuple[int, ...] | None = None
    speech_token_ids: tuple[int, ...] = ()
    audio_artifact_id: str | None = None
    tool_call: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_alignment(self) -> "GenerationRecord":
        if len(self.attention_mask) != len(self.input_ids) + len(self.generated_ids):
            raise ValueError("attention_mask must cover input and generated ids")
        if len(self.action_mask) != len(self.generated_ids):
            raise ValueError("action_mask must align with generated ids")
        if len(self.old_logprobs) != len(self.generated_ids):
            raise ValueError("old_logprobs must align with generated ids")
        if self.position_ids is not None and len(self.position_ids) != len(self.attention_mask):
            raise ValueError("position_ids must align with attention_mask")
        if not all(math.isfinite(value) for value in self.old_logprobs):
            raise ValueError("old_logprobs must be finite")
        if not any(self.action_mask):
            raise ValueError("at least one generated token must be trainable")
        return self


class TrajectoryStepV1(FrozenModel):
    index: int = Field(ge=1)
    generation: GenerationRecord
    action: dict[str, Any]
    observation: dict[str, Any]
    state_before_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    state_after_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    reward_events: tuple[dict[str, Any], ...]
    reward: float
    done: bool


class TrajectoryV1(FrozenModel):
    schema_version: Literal["audioenv.trajectory/v1"] = "audioenv.trajectory/v1"
    experiment_id: str
    environment_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    verifier_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    task_id: str
    episode_id: str
    seed: int
    model_id: str
    checkpoint_id: str
    policy_version: str
    processor_config: dict[str, Any]
    audio_artifacts: tuple[AudioArtifactV1, ...]
    steps: tuple[TrajectoryStepV1, ...]
    terminal_score: float
    truncation_reason: str | None = None
    observability_artifact_id: str | None = None

    @model_validator(mode="after")
    def validate_episode(self) -> "TrajectoryV1":
        if not self.steps:
            if self.truncation_reason is None:
                raise ValueError("trajectory must contain at least one step")
            return self
        expected = list(range(1, len(self.steps) + 1))
        if [step.index for step in self.steps] != expected:
            raise ValueError("trajectory step indices must be contiguous")
        if not self.steps[-1].done and self.truncation_reason is None:
            raise ValueError("trajectory needs a terminal step or truncation reason")
        if any(step.generation.policy_version != self.policy_version for step in self.steps):
            raise ValueError("all generations must use the trajectory policy version")
        known_audio = {item.artifact_id for item in self.audio_artifacts}
        for step in self.steps:
            ref = step.generation.audio_artifact_id
            if ref is not None and ref not in known_audio:
                raise ValueError(f"generation references missing audio artifact: {ref}")
        return self


class TrainerCapabilities(FrozenModel):
    algorithms: tuple[str, ...]
    model_families: tuple[str, ...]
    action_modalities: tuple[str, ...]
    trainable_modules: tuple[str, ...]
    distributed_backends: tuple[str, ...] = ()
    supports_resume: bool = False
    checkpoint_format: str


class ResourceEstimate(FrozenModel):
    provider: str
    device_type: str
    device_count: int = Field(ge=0)
    memory_gb_per_device: float | None = None
    estimated_cost_usd: float | None = None
    cost_basis: str | None = None
    available: bool
    reasons: tuple[str, ...] = ()

