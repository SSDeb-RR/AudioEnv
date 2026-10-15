"""Qwen2.5-Omni boundary for a rollout worker.

Model loading and distributed placement are intentionally caller-owned. This
adapter prevents the environment package from becoming an optimizer or serving
framework. The supplied generate callback must consume raw audio and return an
AudioEnv Action plus policy token metadata.
"""

from __future__ import annotations

from collections.abc import Callable

from .base import ModelAction
from ..spec import AudioObservation


class Qwen25OmniAdapter:
    model_id = "Qwen/Qwen2.5-Omni-7B"

    def __init__(self, generate: Callable[[bytes | None, dict | None], ModelAction]):
        self._generate = generate

    def act(self, observation: AudioObservation) -> ModelAction:
        return self._generate(observation.audio, observation.tool_result)
