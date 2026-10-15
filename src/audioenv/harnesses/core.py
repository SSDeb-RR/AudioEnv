"""Common contracts for environment, model, caller, trainer, and provider harnesses.

The registry is intentionally small and dependency-free. Plugins describe what
they actually support; compatibility is computed from declared capabilities,
not from labels entered in the dashboard.
"""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


HarnessKind = Literal["environment", "model", "caller", "trainer", "provider"]
HarnessStatus = Literal["available", "needs_review", "blocked", "unsupported"]


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class HarnessDescriptor(FrozenModel):
    id: str
    kind: HarnessKind
    display_name: str
    status: HarnessStatus
    status_reason: str
    capabilities: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()
    configuration_schema: dict[str, Any] = Field(default_factory=dict)
    implementation: str | None = None


class SourceAnalysisReport(FrozenModel):
    source_uri: str
    source_kind: Literal["benchmark", "repository", "custom", "unknown"]
    classification: str
    status: HarnessStatus
    detected_files: tuple[str, ...] = ()
    detected_tools: tuple[str, ...] = ()
    task_count: int | None = None
    proposed_adapter: str | None = None
    verifier_source: str | None = None
    execution_mode: Literal["in_process", "oci_sandbox", "not_executable"] = "not_executable"
    generated_files: tuple[str, ...] = ()
    validation_checks: tuple[dict[str, Any], ...] = ()
    review_items: tuple[str, ...] = ()
    environment_id: str | None = None


class ModelOnboardingReport(FrozenModel):
    model_id: str
    source: str
    detected_family: str
    selected_driver: str | None
    status: HarnessStatus
    registration_state: Literal["certified", "conformant", "generated_needs_review", "rejected"] = "generated_needs_review"
    capabilities: tuple[str, ...] = ()
    conformance_checks: tuple[dict[str, Any], ...] = ()
    compatible_trainers: tuple[str, ...] = ()
    generated_files: tuple[str, ...] = ()
    review_items: tuple[str, ...] = ()


class CompatibilityItem(FrozenModel):
    harness_id: str
    compatible: bool
    status: HarnessStatus
    reasons: tuple[str, ...] = ()


class CompatibilityReport(FrozenModel):
    environment_id: str
    model_id: str
    caller_id: str | None = None
    model_capabilities: tuple[str, ...]
    environment_requirements: tuple[str, ...]
    trainers: tuple[CompatibilityItem, ...]
    callers: tuple[CompatibilityItem, ...]


class HarnessRegistry:
    """Runtime-discoverable registry with duplicate protection."""

    def __init__(self) -> None:
        self._items: dict[str, HarnessDescriptor] = {}

    def register(self, descriptor: HarnessDescriptor) -> None:
        if descriptor.id in self._items:
            raise ValueError(f"duplicate harness id: {descriptor.id}")
        self._items[descriptor.id] = descriptor

    def get(self, harness_id: str) -> HarnessDescriptor | None:
        return self._items.get(harness_id)

    def list(self, kind: HarnessKind | None = None) -> tuple[HarnessDescriptor, ...]:
        items = self._items.values()
        if kind is not None:
            items = (item for item in items if item.kind == kind)
        return tuple(sorted(items, key=lambda item: (item.kind, item.id)))

    def discover_entry_points(self, group: str = "audioenv.harnesses") -> tuple[str, ...]:
        loaded: list[str] = []
        for point in entry_points().select(group=group):
            value = point.load()
            descriptors = value() if callable(value) else value
            if isinstance(descriptors, HarnessDescriptor):
                descriptors = (descriptors,)
            for descriptor in descriptors:
                self.register(descriptor)
                loaded.append(descriptor.id)
        return tuple(loaded)


def build_builtin_registry() -> HarnessRegistry:
    registry = HarnessRegistry()
    descriptors = (
        HarnessDescriptor(
            id="audioenv/reference-v2", kind="environment", display_name="Reference V2",
            status="available", status_reason="validated 48-task local environment",
            capabilities=("native-wav-input", "structured-tools", "protected-verifier"),
            implementation="audioenv.reference:build_reference_package",
        ),
        HarnessDescriptor(
            id="audioenv/json-benchmark", kind="environment", display_name="AudioEnv JSON benchmark",
            status="available", status_reason="safe structured manifest importer",
            capabilities=("benchmark-import", "protected-verifier"),
            implementation="audioenv.adapters.json_benchmark:JsonBenchmarkAdapter",
        ),
        HarnessDescriptor(
            id="audioenv/repository-coding", kind="environment", display_name="Repository coding task",
            status="blocked", status_reason="requires an OCI sandbox provider",
            capabilities=("repository-analysis", "test-verifier"), requires=("oci-sandbox",),
        ),
        HarnessDescriptor(
            id="audioenv/small-audio-policy", kind="model", display_name="Small Audio Tool Policy",
            status="available", status_reason="local audio-conditioned proof",
            capabilities=("audio-input", "structured-tools", "token-logprobs", "trainable", "pytorch-state-dict"),
            implementation="audioenv.training.tiny:SmallAudioToolPolicyTrainer",
        ),
        HarnessDescriptor(
            id="audioenv/qwen3-omni", kind="model", display_name="Qwen3 Omni family",
            status="blocked", status_reason="requires CUDA and a local checkpoint",
            capabilities=("audio-input", "audio-output", "structured-tools", "token-logprobs", "trainable", "peft-lora", "distributed"),
            implementation="audioenv.models.qwen3_omni:Qwen3OmniAgentAdapter",
        ),
        HarnessDescriptor(
            id="audioenv/scripted-caller", kind="caller", display_name="Scripted audio caller",
            status="available", status_reason="deterministic fixture caller",
            capabilities=("audio-output", "deterministic"),
            implementation="audioenv.models.caller:ScriptedAudioCallerAdapter",
        ),
        HarnessDescriptor(
            id="audioenv/omni-caller", kind="caller", display_name="Frozen Omni caller",
            status="blocked", status_reason="requires a model driver with audio input and output",
            capabilities=("audio-input", "audio-output", "model-driven", "frozen"),
            requires=("audio-input", "audio-output"),
        ),
        HarnessDescriptor(
            id="audioenv/small-audio-reinforce", kind="trainer", display_name="Small Policy REINFORCE",
            status="available", status_reason="verified local optimizer",
            capabilities=("reinforce", "native-wav-input", "structured-tools", "pytorch-state-dict"),
            requires=("audio-input", "structured-tools", "trainable", "pytorch-state-dict"),
            configuration_schema={"steps": {"type": "integer", "minimum": 1, "default": 40}, "learning_rate": {"type": "number", "exclusiveMinimum": 0, "default": 0.08}},
            implementation="audioenv.training.tiny:SmallAudioToolPolicyTrainer",
        ),
        HarnessDescriptor(
            id="qwen3-omni-thinker-gspo-lora", kind="trainer", display_name="Qwen3-Omni Thinker GSPO LoRA",
            status="blocked", status_reason="GPU acceptance has not run",
            capabilities=("gspo", "audio-input", "structured-tools", "peft-lora", "distributed"),
            requires=("audio-input", "structured-tools", "token-logprobs", "trainable", "peft-lora", "distributed"),
            implementation="audioenv.training.verl_omni:VerlOmniGSPOTrainer",
        ),
        HarnessDescriptor(
            id="qwen3-omni-thinker-grpo-lora", kind="trainer", display_name="Qwen3-Omni Thinker GRPO LoRA",
            status="blocked", status_reason="GPU acceptance has not run",
            capabilities=("grpo", "audio-input", "structured-tools", "peft-lora", "distributed"),
            requires=("audio-input", "structured-tools", "token-logprobs", "trainable", "peft-lora", "distributed"),
            implementation="audioenv.training.verl_omni:VerlOmniGRPOTrainer",
        ),
        HarnessDescriptor(
            id="local-subprocess", kind="provider", display_name="Local subprocess",
            status="available", status_reason="verified CPU execution", capabilities=("cpu", "subprocess"),
        ),
        HarnessDescriptor(
            id="runpod-secure-cloud", kind="provider", display_name="RunPod Secure Cloud",
            status="blocked", status_reason="live REST, SSH, and artifact transport not verified",
            capabilities=("cuda", "distributed", "persistent-volume"),
        ),
        HarnessDescriptor(
            id="modal-sandbox", kind="provider", display_name="Modal Sandbox",
            status="blocked", status_reason="transport contract ready; live credentials and paid acceptance not run",
            capabilities=("cuda", "distributed", "persistent-volume", "gvisor"),
        ),
    )
    for descriptor in descriptors:
        registry.register(descriptor)
    return registry


def resolve_compatibility(
    registry: HarnessRegistry,
    *,
    environment_id: str,
    environment_requirements: tuple[str, ...],
    model_id: str,
    model_capabilities: tuple[str, ...],
    caller_id: str | None = None,
) -> CompatibilityReport:
    capabilities = {_normalize_capability(item) for item in model_capabilities}
    normalized_requirements = {_normalize_capability(item) for item in environment_requirements}
    missing_environment = sorted(normalized_requirements - capabilities)
    trainers: list[CompatibilityItem] = []
    for descriptor in registry.list("trainer"):
        missing = sorted({_normalize_capability(item) for item in descriptor.requires} - capabilities)
        reasons = tuple([*(f"model lacks {item}" for item in missing), *(f"environment requires {item}" for item in missing_environment)])
        compatible = not reasons and descriptor.status == "available"
        if descriptor.status != "available":
            reasons = (*reasons, descriptor.status_reason)
        trainers.append(CompatibilityItem(
            harness_id=descriptor.id, compatible=compatible, status=descriptor.status, reasons=reasons,
        ))
    callers: list[CompatibilityItem] = []
    for descriptor in registry.list("caller"):
        missing = sorted({_normalize_capability(item) for item in descriptor.requires} - capabilities)
        reasons = tuple(f"model lacks {item}" for item in missing)
        compatible = not reasons and descriptor.status == "available"
        if descriptor.status != "available":
            reasons = (*reasons, descriptor.status_reason)
        callers.append(CompatibilityItem(
            harness_id=descriptor.id, compatible=compatible, status=descriptor.status, reasons=reasons,
        ))
    return CompatibilityReport(
        environment_id=environment_id,
        model_id=model_id,
        caller_id=caller_id,
        model_capabilities=tuple(sorted(capabilities)),
        environment_requirements=tuple(sorted(normalized_requirements)),
        trainers=tuple(trainers),
        callers=tuple(callers),
    )


def _normalize_capability(value: str) -> str:
    aliases = {
        "native-wav-input": "audio-input",
        "structured-tool-calls": "structured-tools",
        "token-log-probabilities": "token-logprobs",
    }
    normalized = value.casefold().replace("_", "-")
    return aliases.get(normalized, normalized)
