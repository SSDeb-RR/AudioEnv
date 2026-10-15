from pathlib import Path

from audioenv.contracts import GenerationRecord, ModelCapabilities, ProbeResult
from audioenv.models import (
    ConformanceFixture,
    ModelConformanceRunner,
    OptimizerStepEvidence,
)
from audioenv.spec import AudioObservation, ToolCallAction


class _ConformantDriver:
    driver_id = "test/conformant-omni"
    model_id = "lab/conformant-omni"

    @property
    def capabilities(self):
        return ModelCapabilities(
            audio_input=True, audio_output=True, structured_tools=True,
            token_logprobs=True, trainable=True,
            trainable_modules=("q_proj",), checkpoint_format="test-lora",
        )

    def probe(self):
        return ProbeResult(
            available=True, model_id=self.model_id, capabilities=self.capabilities,
            device="cpu", revision="fixture-v1",
        )

    def load(self): pass
    def reset_conversation(self): pass

    def generate_step(self, observation):
        assert observation.audio
        record = GenerationRecord(
            generation_id="g1", model_id=self.model_id, model_revision="fixture-v1",
            policy_version="p1", semantic_text='{"tool_call":{"name":"lookup","arguments":{}}}',
            input_ids=(1, 2), generated_ids=(3, 4), attention_mask=(1, 1, 1, 1),
            action_mask=(1, 1), old_logprobs=(-0.2, -0.3),
            position_ids=((0, 1, 2, 3), (0, 0, 1, 1)),
            tool_call={"tool_call": {"name": "lookup", "arguments": {}}},
        )
        return ToolCallAction("lookup", {}), record

    def configure_lora(self, **kwargs):
        assert kwargs["targets"]
        return ("thinker.q_proj.lora_A", "thinker.q_proj.lora_B")

    def trainable_parameter_names(self): return ("thinker.q_proj.lora_A", "thinker.q_proj.lora_B")
    def parameter_hash(self, **kwargs): return "a" * 64
    def save_adapter(self, destination): return destination
    def load_adapter(self, source): pass

    def conformance_optimizer_step(self, record):
        assert record.generated_ids
        return OptimizerStepEvidence(
            before_hash="a" * 64, after_hash="b" * 64,
            gradient_parameter_names=("thinker.q_proj.lora_A", "thinker.q_proj.lora_B"),
            optimizer_state_changed=True,
        )

    def conformance_checkpoint_roundtrip(self, destination: Path, fixture):
        destination.mkdir(exist_ok=True)
        return bool(fixture.observation.audio)

    def conformance_deterministic_probe(self, fixture):
        return bool(fixture.observation.audio)


def test_conformant_omni_driver_passes_every_executable_gate():
    report = ModelConformanceRunner().run(
        _ConformantDriver(), ConformanceFixture(AudioObservation(audio=b"RIFF-fixture")),
    )
    assert report.registered is True
    assert report.registration_state == "conformant"
    assert {item.name for item in report.checks} == set(ModelConformanceRunner.check_names)
    assert all(item.passed for item in report.checks)


def test_multimodal_position_channels_reject_misaligned_widths():
    try:
        GenerationRecord(
            generation_id="bad", model_id="m", model_revision="r", policy_version="p",
            semantic_text="x", input_ids=(1,), generated_ids=(2,), attention_mask=(1, 1),
            action_mask=(1,), old_logprobs=(-1.0,), position_ids=((0,), (0, 1)),
        )
    except ValueError as exc:
        assert "position channels" in str(exc)
    else:
        raise AssertionError("misaligned multimodal positions were accepted")
