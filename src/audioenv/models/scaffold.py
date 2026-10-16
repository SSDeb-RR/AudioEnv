"""Deterministic, review-gated SDK package for an unknown Omni family."""

from __future__ import annotations

import io
import json
import re
import zipfile
from hashlib import sha256
from typing import Any


def build_driver_scaffold(
    model_id: str, source: str, detected_family: str,
    model_config: dict[str, Any] | None = None,
) -> bytes:
    family_slug = re.sub(r"[^a-z0-9]+", "_", detected_family.casefold()).strip("_") or "unknown_omni"
    adapter_plan = _adapter_plan(model_config or {})
    manifest = {
        "schema_version": "audioenv.model-driver-scaffold/v2",
        "model_id": model_id,
        "source": source,
        "detected_family": detected_family,
        "module": f"audioenv_{family_slug}_driver.driver:GeneratedOmniDriver",
        "registration_state": "generated_needs_review",
        "adapter_plan_sha256": sha256(
            json.dumps(adapter_plan, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "required_conformance": [
            "dependency_probe", "native_audio_inference", "structured_action",
            "token_logprob_alignment", "multimodal_positions", "mask_isolation",
            "declared_gradients", "lora_update", "checkpoint_reload", "deterministic_probe",
        ],
    }
    package = f"audioenv_{family_slug}_driver"
    driver = _DRIVER_TEMPLATE.replace("__MODEL_ID__", repr(model_id))
    test = _TEST_TEMPLATE.replace("__PACKAGE__", package)
    files = {
        "manifest.json": json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        "adapter_spec.json": json.dumps(adapter_plan, sort_keys=True, indent=2) + "\n",
        f"{package}/__init__.py": "from .driver import GeneratedOmniDriver\n",
        f"{package}/driver.py": driver,
        "tests/test_driver.py": test,
        "README.md": _README,
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, content)
    return buffer.getvalue()


def _adapter_plan(config: dict[str, Any]) -> dict[str, Any]:
    architectures = config.get("architectures")
    auto_map = config.get("auto_map")
    safe_auto_map = {
        str(key): str(value)
        for key, value in auto_map.items()
        if isinstance(auto_map, dict) and isinstance(key, str) and isinstance(value, str)
    } if isinstance(auto_map, dict) else {}
    return {
        "schema_version": "audioenv.omni-adapter-plan/v1",
        "model_type": str(config.get("model_type") or "unknown")[:200],
        "architectures": [str(value)[:200] for value in architectures[:20]]
        if isinstance(architectures, list) else [],
        "processor_class": str(config.get("processor_class") or "AutoProcessor")[:200],
        "auto_map": safe_auto_map,
        "trust_remote_code_required": bool(safe_auto_map),
        "implementation_order": [
            "load processor and model without network access",
            "construct native waveform inputs with the model processor",
            "return generated token ids, positions, response mask, and token log-probabilities",
            "parse constrained structured tool actions",
            "render speech from the same semantic generation record",
            "inject LoRA only into declared trainable modules",
            "implement content-hash-bound adapter save and reload",
            "pass every isolated AudioEnv conformance gate",
        ],
    }


_DRIVER_TEMPLATE = '''"""Generated skeleton. Unavailable until every method is implemented and tested."""
from audioenv.contracts import ModelCapabilities, ProbeResult


class GeneratedOmniDriver:
    driver_id = "generated/review-required"
    model_id = __MODEL_ID__

    @property
    def capabilities(self):
        return ModelCapabilities(audio_input=False, audio_output=False, structured_tools=False,
                                 token_logprobs=False, trainable=False)

    def probe(self):
        return ProbeResult(available=False, model_id=self.model_id, capabilities=self.capabilities,
                           reasons=("generated adapter requires implementation and conformance",))

    def load(self): raise NotImplementedError
    def reset_conversation(self): raise NotImplementedError
    def generate_step(self, observation): raise NotImplementedError
    def configure_lora(self, *, rank, alpha, targets): raise NotImplementedError
    def trainable_parameter_names(self): raise NotImplementedError
    def parameter_hash(self, *, trainable_only=True): raise NotImplementedError
    def save_adapter(self, destination): raise NotImplementedError
    def load_adapter(self, source): raise NotImplementedError
    def conformance_optimizer_step(self, record): raise NotImplementedError
    def conformance_checkpoint_roundtrip(self, destination, fixture): raise NotImplementedError
    def conformance_deterministic_probe(self, fixture): raise NotImplementedError
'''

_TEST_TEMPLATE = '''from __PACKAGE__.driver import GeneratedOmniDriver


def test_generated_driver_stays_unavailable_before_implementation():
    driver = GeneratedOmniDriver()
    assert driver.probe().available is False
'''

_README = """# Generated AudioEnv Omni driver

This package is a review-required SDK scaffold, not a working model adapter.
Implement the driver methods, declare only measured capabilities, then run the
AudioEnv conformance suite in an isolated build environment. Registration is
forbidden until every manifest gate passes.
"""
