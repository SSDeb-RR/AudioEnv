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
    position_ids: tuple[int, ...] | tuple[tuple[int, ...], ...] | None = None
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
        if self.position_ids is not None:
            if self.position_ids and isinstance(self.position_ids[0], tuple):
                widths = {len(channel) for channel in self.position_ids}
                if widths != {len(self.attention_mask)}:
                    raise ValueError("multimodal position channels must align with attention_mask")
            elif len(self.position_ids) != len(self.attention_mask):
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


class SplitManifestV1(FrozenModel):
    schema_version: Literal["audioenv.split/v1"] = "audioenv.split/v1"
    environment_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]
    seeds: dict[str, tuple[int, ...]]
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @staticmethod
    def hash_payload(payload: dict[str, Any]) -> str:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return sha256(encoded).hexdigest()

    @model_validator(mode="after")
    def validate_split(self) -> "SplitManifestV1":
        groups = [set(self.train), set(self.validation), set(self.test)]
        if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
            raise ValueError("task ids may appear in exactly one split")
        payload = self.model_dump(mode="json")
        payload.pop("content_hash", None)
        if self.hash_payload(payload) != self.content_hash:
            raise ValueError("split manifest content_hash does not match")
        return self


class ExperimentConfigV1(FrozenModel):
    schema_version: Literal["audioenv.experiment/v1"] = "audioenv.experiment/v1"
    project_id: str
    environment_id: str
    environment_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    verifier_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    split_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    agent_model_id: str
    caller_model_id: str | None = None
    trainer_id: str
    provider_id: str
    hyperparameters: dict[str, Any]
    reward_version: str
    seeds: tuple[int, ...]

    @property
    def content_hash(self) -> str:
        return sha256(
            json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


class EvaluationProtocolV1(FrozenModel):
    schema_version: Literal["audioenv.evaluation-protocol/v1"] = "audioenv.evaluation-protocol/v1"
    environment_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    verifier_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    split_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    split: Literal["validation", "test"]
    caller_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    audio_pipeline_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    sampling_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoint_id: str
    seeds: tuple[int, ...]

    @property
    def compatibility_hash(self) -> str:
        data = self.model_dump(mode="json")
        data.pop("checkpoint_id")
        return sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    @property
    def protocol_hash(self) -> str:
        return sha256(
            json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


class EvaluationSampleV1(FrozenModel):
    task_id: str
    audio_profile_id: str
    seed: int
    passed: bool
    reward: float
    safety_violations: int = Field(default=0, ge=0)
    invalid_calls: int = Field(default=0, ge=0)
    repeated_failed_calls: int = Field(default=0, ge=0)
    failure_category: str | None = None


class EvaluationResultV1(FrozenModel):
    schema_version: Literal["audioenv.evaluation-result/v1"] = "audioenv.evaluation-result/v1"
    protocol: EvaluationProtocolV1
    samples: tuple[EvaluationSampleV1, ...]
    success_rate: float = Field(ge=0, le=1)
    success_ci_low: float = Field(ge=0, le=1)
    success_ci_high: float = Field(ge=0, le=1)
    mean_reward: float
    reward_components: dict[str, float] = Field(default_factory=dict)
    failure_counts: dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_samples(self) -> "EvaluationResultV1":
        if not self.samples:
            raise ValueError("evaluation result requires samples")
        return self


class ComparisonV1(FrozenModel):
    schema_version: Literal["audioenv.comparison/v1"] = "audioenv.comparison/v1"
    baseline_protocol_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    trained_protocol_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    compatibility_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    compatible: bool
    incompatibilities: tuple[str, ...] = ()
    paired_sample_count: int = Field(ge=0)
    baseline_success_rate: float | None = None
    trained_success_rate: float | None = None
    paired_delta: float | None = None
    paired_ci_low: float | None = None
    paired_ci_high: float | None = None


class TrainerDescriptor(FrozenModel):
    id: str
    display_name: str
    capabilities: TrainerCapabilities
    configuration_schema: dict[str, Any]
    required_rollout_fields: tuple[str, ...]
    status: Literal["available", "blocked", "unavailable"]
    status_reason: str | None = None


class ExecutionProviderDescriptor(FrozenModel):
    id: str
    display_name: str
    local: bool
    supports_quote: bool
    supports_cancel: bool
    configuration_schema: dict[str, Any]
    prerequisites: tuple[str, ...]
    status: Literal["available", "blocked", "unavailable"]
    status_reason: str | None = None


class SandboxPolicyV1(FrozenModel):
    network: Literal["off", "bootstrap-only"] = "off"
    run_as_non_root: bool = True
    read_only_base: bool = True
    ephemeral_workspace: bool = True
    host_socket: bool = False
    inherit_secrets: bool = False
    cpu_limit: float = Field(default=2.0, gt=0)
    memory_mb: int = Field(default=4096, gt=0)
    pid_limit: int = Field(default=256, gt=0)
    timeout_seconds: int = Field(default=600, gt=0)


class FoundryJobV1(FrozenModel):
    schema_version: Literal["audioenv.foundry-job/v1"] = "audioenv.foundry-job/v1"
    id: str
    source_uri: str
    generator: Literal["deterministic", "codex", "repo2rlenv", "tau2"]
    status: Literal["draft", "analyzing", "generating", "validating", "completed", "failed"]
    sandbox_policy: SandboxPolicyV1
    logs: tuple[str, ...] = ()
    generated_artifact_ids: tuple[str, ...] = ()
    validation_report: dict[str, Any] | None = None
    promotion: Literal["validated", "generated_needs_review", "unsupported"] | None = None
    failure_reason: str | None = None


class PromotionReportV1(FrozenModel):
    schema_version: Literal["audioenv.promotion/v1"] = "audioenv.promotion/v1"
    environment_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    verifier_hash_before: str = Field(pattern=r"^[0-9a-f]{64}$")
    verifier_hash_after: str = Field(pattern=r"^[0-9a-f]{64}$")
    checks: dict[str, bool]
    promoted: bool
    missing_or_failed: tuple[str, ...] = ()
    evidence_artifact_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    worker_id: str | None = None

    @model_validator(mode="after")
    def enforce_hard_gates(self) -> "PromotionReportV1":
        required = {
            "sandbox_build", "oracle_pass", "noop_fail", "adversarial_fail",
            "determinism", "state_isolation", "leakage", "reward_integrity",
        }
        failed = sorted(name for name in required if not self.checks.get(name, False))
        if self.verifier_hash_before != self.verifier_hash_after:
            failed.append("verifier_immutability")
        if self.promoted and failed:
            raise ValueError("promotion requires every hard gate")
        if self.promoted != (not failed):
            raise ValueError("promoted must reflect the hard-gate result")
        if tuple(failed) != self.missing_or_failed:
            raise ValueError("missing_or_failed must list the failed hard gates")
        return self

    @classmethod
    def evaluate(
        cls, *, environment_hash: str, verifier_hash_before: str,
        verifier_hash_after: str, checks: dict[str, bool],
        evidence_artifact_id: str | None = None, worker_id: str | None = None,
    ) -> "PromotionReportV1":
        required = {
            "sandbox_build", "oracle_pass", "noop_fail", "adversarial_fail",
            "determinism", "state_isolation", "leakage", "reward_integrity",
        }
        failed = sorted(name for name in required if not checks.get(name, False))
        if verifier_hash_before != verifier_hash_after:
            failed.append("verifier_immutability")
        return cls(
            environment_hash=environment_hash,
            verifier_hash_before=verifier_hash_before,
            verifier_hash_after=verifier_hash_after,
            checks=checks, promoted=not failed, missing_or_failed=tuple(failed),
            evidence_artifact_id=evidence_artifact_id, worker_id=worker_id,
        )


class PromotionEvidenceV1(FrozenModel):
    schema_version: Literal["audioenv.promotion-evidence/v1"] = "audioenv.promotion-evidence/v1"
    environment_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    verifier_hash_before: str = Field(pattern=r"^[0-9a-f]{64}$")
    verifier_hash_after: str = Field(pattern=r"^[0-9a-f]{64}$")
    checks: dict[str, bool]
    worker_id: str
    issued_at: str
    nonce: str
    key_id: str = Field(pattern=r"^[0-9a-f]{16}$")
    signature: str = Field(pattern=r"^[0-9a-f]{64}$")


class ConformanceCheckV1(FrozenModel):
    name: str
    passed: bool
    evidence: dict[str, Any] = Field(default_factory=dict)
    reason: str | None = None


class ModelConformanceReportV1(FrozenModel):
    schema_version: Literal["audioenv.model-conformance/v1"] = "audioenv.model-conformance/v1"
    model_id: str
    driver_id: str
    checks: tuple[ConformanceCheckV1, ...]
    registered: bool
    registration_state: Literal["certified", "conformant", "generated_needs_review", "rejected"] = "generated_needs_review"
    generated_artifact_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def registration_requires_all_checks(self) -> "ModelConformanceReportV1":
        if self.registered and (not self.checks or not all(item.passed for item in self.checks)):
            raise ValueError("a model driver can register only after every conformance check passes")
        if self.registered != (self.registration_state in {"certified", "conformant"}):
            raise ValueError("registration state must agree with registered")
        return self


class OnlineRLTrainerConfigV1(FrozenModel):
    schema_version: Literal["audioenv.online-rl/v1"] = "audioenv.online-rl/v1"
    algorithm: Literal["gspo", "grpo"]
    model_id: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    group_size: int = Field(default=4, ge=2)
    learning_rate: float = Field(default=1e-6, gt=0)
    max_steps: int = Field(default=1, ge=1)
    lora_rank: int = Field(default=32, ge=1)
    lora_alpha: int = Field(default=64, ge=1)
    lora_targets: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    temperature: float = Field(default=1.0, gt=0)
    top_p: float = Field(default=1.0, gt=0, le=1)
    terminal_reward_weight: float = Field(default=1.0, ge=0)
    process_reward_weight: float = Field(default=1.0, ge=0)
    checkpoint_interval: int = Field(default=20, ge=1)
    evaluation_interval: int = Field(default=20, ge=1)
    resume_from: str | None = None


class MultimodalFeatureRefV1(FrozenModel):
    kind: Literal["audio", "image", "video"]
    artifact_id: str
    start_token: int = Field(ge=0)
    end_token: int = Field(gt=0)

    @model_validator(mode="after")
    def ordered_span(self) -> "MultimodalFeatureRefV1":
        if self.end_token <= self.start_token:
            raise ValueError("multimodal feature span must be ordered")
        return self


class TrainingTurnV2(FrozenModel):
    index: int = Field(ge=1)
    input_ids: tuple[int, ...]
    generated_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    response_mask: tuple[int, ...]
    position_ids: tuple[int, ...]
    rollout_logprobs: tuple[float, ...]
    multimodal_features: tuple[MultimodalFeatureRefV1, ...]
    action: dict[str, Any]
    reward_events: tuple[dict[str, Any], ...]
    done: bool

    @model_validator(mode="after")
    def validate_training_alignment(self) -> "TrainingTurnV2":
        total = len(self.input_ids) + len(self.generated_ids)
        if len(self.attention_mask) != total or len(self.position_ids) != total:
            raise ValueError("attention and position data must span prompt plus response")
        if len(self.response_mask) != len(self.generated_ids):
            raise ValueError("response mask must align with generated ids")
        if len(self.rollout_logprobs) != len(self.generated_ids):
            raise ValueError("rollout log-probabilities must align with generated ids")
        if any(value not in {0, 1} for value in self.response_mask) or not any(self.response_mask):
            raise ValueError("response mask must contain at least one trainable token")
        if not all(math.isfinite(value) for value in self.rollout_logprobs):
            raise ValueError("rollout log-probabilities must be finite")
        return self


class TrajectoryV2(FrozenModel):
    schema_version: Literal["audioenv.trajectory/v2"] = "audioenv.trajectory/v2"
    experiment_id: str
    environment_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    verifier_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    reward_version: str
    task_id: str
    episode_id: str
    seed: int
    model_id: str
    checkpoint_id: str
    policy_version: str
    processor_config: dict[str, Any]
    audio_artifacts: tuple[AudioArtifactV1, ...]
    turns: tuple[TrainingTurnV2, ...]
    terminal_reward: float
    process_reward: float
    done: bool
    truncation_reason: str | None = None

    @model_validator(mode="after")
    def validate_episode(self) -> "TrajectoryV2":
        if not self.turns:
            raise ValueError("trajectory requires at least one training turn")
        if tuple(item.index for item in self.turns) != tuple(range(1, len(self.turns) + 1)):
            raise ValueError("training turn indices must be contiguous")
        if not self.done and self.truncation_reason is None:
            raise ValueError("non-terminal trajectories require a truncation reason")
        known_audio = {item.artifact_id for item in self.audio_artifacts}
        if any(
            feature.artifact_id not in known_audio
            for turn in self.turns for feature in turn.multimodal_features
            if feature.kind == "audio"
        ):
            raise ValueError("multimodal feature references missing audio")
        return self


class GPUExecutionPlanV1(FrozenModel):
    schema_version: Literal["audioenv.gpu-execution/v1"] = "audioenv.gpu-execution/v1"
    provider: Literal["local-cuda", "runpod-secure-cloud"]
    image: str
    image_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    gpu_type: str
    gpu_count: int = Field(ge=1)
    memory_gb_per_gpu: int = Field(ge=1)
    storage_gb: int = Field(ge=1)
    secret_names: tuple[str, ...] = ()
    max_runtime_hours: float = Field(gt=0)
    max_total_budget_usd: float = Field(ge=0)
    hourly_price_usd: float | None = Field(default=None, ge=0)
    confirmation_name: str | None = None

    @property
    def maximum_cost_usd(self) -> float | None:
        return None if self.hourly_price_usd is None else self.hourly_price_usd * self.max_runtime_hours


class GPUWorkerExperimentV1(FrozenModel):
    """Portable input consumed identically by local and cloud GPU workers."""

    schema_version: Literal["audioenv.gpu-worker-experiment/v1"] = "audioenv.gpu-worker-experiment/v1"
    experiment_id: str
    run_name: str
    backend: Literal["tiny-cpu-proof", "qwen3-omni-verl"]
    trainer: Literal[
        "audioenv/small-audio-reinforce",
        "qwen3-omni-thinker-gspo-lora",
        "qwen3-omni-thinker-grpo-lora",
    ]
    algorithm: Literal["reinforce", "gspo", "grpo"]
    environment_manifest: str | None = None
    model_path: str | None = None
    caller_model_path: str | None = None
    rollout_dataset_path: str | None = None
    verl_checkout: str | None = None
    output_dir: str
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    dependency_pins: dict[str, str] = Field(default_factory=dict)
    resume_from: str | None = None
    cancellation_file: str | None = None

    @model_validator(mode="after")
    def validate_backend(self) -> "GPUWorkerExperimentV1":
        if self.backend == "tiny-cpu-proof":
            if self.trainer != "audioenv/small-audio-reinforce" or self.algorithm != "reinforce":
                raise ValueError("tiny-cpu-proof requires the local REINFORCE trainer")
        else:
            required = {
                "environment_manifest": self.environment_manifest,
                "model_path": self.model_path,
                "caller_model_path": self.caller_model_path,
                "rollout_dataset_path": self.rollout_dataset_path,
                "verl_checkout": self.verl_checkout,
            }
            missing = sorted(name for name, value in required.items() if not value)
            if missing:
                raise ValueError("qwen3-omni-verl requires " + ", ".join(missing))
            expected = f"qwen3-omni-thinker-{self.algorithm}-lora"
            if self.trainer != expected or self.algorithm not in {"gspo", "grpo"}:
                raise ValueError("Qwen backend trainer and algorithm do not match")
        return self

    @property
    def content_hash(self) -> str:
        return sha256(
            json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


class GPUWorkerEventV1(FrozenModel):
    schema_version: Literal["audioenv.gpu-worker-event/v1"] = "audioenv.gpu-worker-event/v1"
    sequence: int = Field(ge=1)
    experiment_id: str
    type: Literal[
        "worker.started", "preflight.completed", "training.started", "metric",
        "checkpoint.saved", "evaluation.completed", "artifact.manifest",
        "worker.cancelled", "worker.failed", "worker.completed",
    ]
    timestamp: str
    payload: dict[str, Any] = Field(default_factory=dict)

