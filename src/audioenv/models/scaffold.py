"""Deterministic, review-gated SDK package for an unknown Omni family."""

from __future__ import annotations

import io
import json
import re
import zipfile


def build_driver_scaffold(model_id: str, source: str, detected_family: str) -> bytes:
    family_slug = re.sub(r"[^a-z0-9]+", "_", detected_family.casefold()).strip("_") or "unknown_omni"
    manifest = {
        "schema_version": "audioenv.model-driver-scaffold/v2",
        "model_id": model_id,
        "source": source,
        "detected_family": detected_family,
        "module": f"audioenv_{family_slug}_driver.driver:GeneratedOmniDriver",
        "registration_state": "generated_needs_review",
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
