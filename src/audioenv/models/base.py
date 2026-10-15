"""Model adapter contract used by rollout workers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ..contracts import GenerationRecord, ModelCapabilities, ProbeResult
from ..spec import Action, AudioObservation, JSONValue


@dataclass(frozen=True, slots=True)
class ModelAction:
    action: Action
    token_ids: tuple[int, ...]
    logprobs: tuple[float, ...]
    metadata: dict[str, JSONValue]


class NativeAudioModelAdapter(Protocol):
    model_id: str

    def act(self, observation: AudioObservation) -> ModelAction: ...


class V1ModelAdapter(Protocol):
    """Capability-declaring model boundary used by V1 workers."""

    model_id: str

    @property
    def capabilities(self) -> ModelCapabilities: ...

    def probe(self) -> ProbeResult: ...

    def generate_step(self, observation: AudioObservation) -> tuple[Action, GenerationRecord]: ...

    def save_adapter(self, destination: Path) -> Path: ...

    def load_adapter(self, source: Path) -> None: ...
