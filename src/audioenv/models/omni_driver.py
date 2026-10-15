"""Architecture-family driver and executable conformance suite for Omni models."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import tempfile
from typing import Any, Protocol

from ..contracts import (
    ConformanceCheckV1,
    GenerationRecord,
    ModelCapabilities,
    ModelConformanceReportV1,
    ProbeResult,
)
from ..spec import Action, AudioObservation


class OmniModelDriver(Protocol):
    driver_id: str
    model_id: str

    @property
    def capabilities(self) -> ModelCapabilities: ...
    def probe(self) -> ProbeResult: ...
    def load(self) -> None: ...
    def reset_conversation(self) -> None: ...
    def generate_step(self, observation: AudioObservation) -> tuple[Action, GenerationRecord]: ...
    def configure_lora(self, *, rank: int, alpha: int, targets: tuple[str, ...]) -> tuple[str, ...]: ...
    def trainable_parameter_names(self) -> tuple[str, ...]: ...
    def parameter_hash(self, *, trainable_only: bool = True) -> str: ...
    def save_adapter(self, destination: Path) -> Path: ...
    def load_adapter(self, source: Path) -> None: ...
    def conformance_optimizer_step(self, record: GenerationRecord) -> "OptimizerStepEvidence": ...
    def conformance_checkpoint_roundtrip(self, destination: Path, fixture: "ConformanceFixture") -> bool: ...
    def conformance_deterministic_probe(self, fixture: "ConformanceFixture") -> bool: ...


@dataclass(frozen=True, slots=True)
class OptimizerStepEvidence:
    before_hash: str
    after_hash: str
    gradient_parameter_names: tuple[str, ...]
    optimizer_state_changed: bool


class Qwen3OmniDriver:
    """One Qwen implementation reused by rollout, training, and evaluation."""

    driver_id = "audioenv/qwen3-omni-v1"

    def __init__(self, adapter):
        self.adapter = adapter
        self.model_id = adapter.model_id

    @property
    def capabilities(self) -> ModelCapabilities:
        return self.adapter.capabilities

    def probe(self) -> ProbeResult:
        return self.adapter.probe()

    def load(self) -> None:
        self.adapter.load()

    def reset_conversation(self) -> None:
        self.adapter.reset_conversation()

    def generate_step(self, observation: AudioObservation):
        return self.adapter.generate_step(observation)

    def configure_lora(self, *, rank: int, alpha: int, targets: tuple[str, ...]) -> tuple[str, ...]:
        if self.adapter._model is None:  # noqa: SLF001 - driver owns adapter lifecycle
            raise RuntimeError("load the Qwen driver before configuring LoRA")
        if rank < 1 or alpha < 1 or not targets:
            raise ValueError("LoRA rank, alpha, and targets are required")
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as exc:  # pragma: no cover - GPU extra
            raise RuntimeError("install peft in the Qwen training image") from exc
        for parameter in self.adapter._model.parameters():
            parameter.requires_grad = False
        config = LoraConfig(
            r=rank, lora_alpha=alpha, target_modules=list(targets),
            lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        )
        self.adapter._model = get_peft_model(self.adapter._model, config)
        names = self.trainable_parameter_names()
        if not names or any("lora_" not in name for name in names):
            raise RuntimeError("only LoRA parameters may remain trainable")
        return names

    def trainable_parameter_names(self) -> tuple[str, ...]:
        model = self.adapter._model  # noqa: SLF001
        if model is None:
            return ()
        return tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad)

    def parameter_hash(self, *, trainable_only: bool = True) -> str:
        model = self.adapter._model  # noqa: SLF001
        if model is None:
            raise RuntimeError("model is not loaded")
        digest = sha256()
        for name, parameter in sorted(model.named_parameters()):
            if trainable_only and not parameter.requires_grad:
                continue
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def save_adapter(self, destination: Path) -> Path:
        return self.adapter.save_adapter(destination)

    def load_adapter(self, source: Path) -> None:
        self.adapter.load_adapter(source)

    def conformance_optimizer_step(self, record: GenerationRecord) -> OptimizerStepEvidence:
        del record
        raise RuntimeError("Qwen optimizer-step conformance must run in the pinned CUDA worker")

    def conformance_checkpoint_roundtrip(self, destination: Path, fixture: "ConformanceFixture") -> bool:
        del destination, fixture
        raise RuntimeError("Qwen checkpoint conformance must run in the pinned CUDA worker")

    def conformance_deterministic_probe(self, fixture: "ConformanceFixture") -> bool:
        del fixture
        raise RuntimeError("Qwen deterministic probe must run in the pinned CUDA worker")


@dataclass(frozen=True, slots=True)
class ConformanceFixture:
    observation: AudioObservation
    require_audio_output: bool = True


class ModelConformanceRunner:
    """Register a driver only after evidence-producing executable checks."""

    check_names = (
        "dependency_probe", "native_audio_inference", "structured_action",
        "token_logprob_alignment", "multimodal_positions", "mask_isolation",
        "declared_gradients", "lora_update", "checkpoint_reload", "deterministic_probe",
    )

    def run(self, driver: OmniModelDriver, fixture: ConformanceFixture) -> ModelConformanceReportV1:
        checks: list[ConformanceCheckV1] = []
        probe = driver.probe()
        checks.append(ConformanceCheckV1(
            name="dependency_probe", passed=probe.available,
            evidence={"device": probe.device, "revision": probe.revision},
            reason="; ".join(probe.reasons) or None,
        ))
        if not probe.available:
            checks.extend(ConformanceCheckV1(
                name=name, passed=False, reason="dependency probe failed",
            ) for name in self.check_names[1:])
            return ModelConformanceReportV1(
                model_id=driver.model_id, driver_id=driver.driver_id,
                checks=tuple(checks), registered=False, registration_state="rejected",
            )
        try:
            driver.load()
            driver.reset_conversation()
            action, record = driver.generate_step(fixture.observation)
            checks.append(ConformanceCheckV1(
                name="native_audio_inference", passed=bool(fixture.observation.audio),
                evidence={"generation_id": record.generation_id},
            ))
            checks.append(ConformanceCheckV1(
                name="structured_action", passed=getattr(action, "kind", None) in {"speak", "tool_call"},
                evidence={"action_kind": getattr(action, "kind", None)},
            ))
            aligned = len(record.generated_ids) == len(record.old_logprobs) == len(record.action_mask)
            checks.append(ConformanceCheckV1(name="token_logprob_alignment", passed=aligned))
            positions = record.position_ids
            position_width = 0
            if positions and isinstance(positions[0], tuple):
                position_width = len(positions[0])
            elif positions:
                position_width = len(positions)
            checks.append(ConformanceCheckV1(
                name="multimodal_positions",
                passed=position_width == len(record.attention_mask),
                evidence={"position_width": position_width, "sequence_width": len(record.attention_mask)},
                reason=None if position_width == len(record.attention_mask) else "position data is missing or misaligned",
            ))
            checks.append(ConformanceCheckV1(
                name="mask_isolation", passed=aligned and all(value in {0, 1} for value in record.action_mask)
                and any(record.action_mask),
            ))
            names = driver.configure_lora(
                rank=2, alpha=4, targets=("q_proj", "k_proj", "v_proj", "o_proj"),
            )
            declared = bool(names) and all("lora_" in name for name in names)
            checks.append(ConformanceCheckV1(
                name="declared_gradients", passed=declared, evidence={"parameters": list(names)},
            ))
            update = driver.conformance_optimizer_step(record)
            gradients_ok = bool(update.gradient_parameter_names) and all(
                name in names for name in update.gradient_parameter_names
            )
            checks.append(ConformanceCheckV1(
                name="lora_update",
                passed=update.before_hash != update.after_hash and update.optimizer_state_changed and gradients_ok,
                evidence={
                    "before_hash": update.before_hash, "after_hash": update.after_hash,
                    "gradient_parameter_names": list(update.gradient_parameter_names),
                    "optimizer_state_changed": update.optimizer_state_changed,
                },
            ))
            with tempfile.TemporaryDirectory(prefix="audioenv-model-conformance-") as directory:
                reloaded = driver.conformance_checkpoint_roundtrip(Path(directory), fixture)
            checks.append(ConformanceCheckV1(name="checkpoint_reload", passed=reloaded))
            deterministic = driver.conformance_deterministic_probe(fixture)
            checks.append(ConformanceCheckV1(name="deterministic_probe", passed=deterministic))
        except Exception as exc:
            present = {item.name for item in checks}
            checks.extend(ConformanceCheckV1(
                name=name, passed=False, reason=str(exc),
            ) for name in self.check_names if name not in present)
        registered = all(item.passed for item in checks)
        return ModelConformanceReportV1(
            model_id=driver.model_id, driver_id=driver.driver_id,
            checks=tuple(checks), registered=registered,
            registration_state="conformant" if registered else "rejected",
        )
