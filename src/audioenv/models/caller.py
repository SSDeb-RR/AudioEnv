"""Frozen caller adapter contracts and deterministic testing caller."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..audio import AudioRenderer, DeterministicWavFixtureRenderer
from ..contracts import ModelCapabilities, ProbeResult
from ..spec import AudioProfile


@dataclass(frozen=True, slots=True)
class CallerTurn:
    audio: bytes
    source: str
    turn_index: int


class CallerAdapter(Protocol):
    model_id: str

    @property
    def capabilities(self) -> ModelCapabilities: ...

    def probe(self) -> ProbeResult: ...

    def reset(self, seed: int) -> CallerTurn: ...

    def respond(self, agent_audio: bytes) -> CallerTurn | None: ...


class ScriptedAudioCallerAdapter:
    """Deterministic fixture caller; explicitly not model-driven speech."""

    model_id = "audioenv/scripted-audio-caller"

    def __init__(
        self,
        private_turns: tuple[str, ...],
        profile: AudioProfile | None = None,
        renderer: AudioRenderer | None = None,
    ):
        if not private_turns:
            raise ValueError("scripted caller requires at least one turn")
        self.private_turns = private_turns
        self.profile = profile or AudioProfile()
        self.renderer = renderer or DeterministicWavFixtureRenderer()
        self._seed = 0
        self._index = 0

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            audio_input=False,
            audio_output=True,
            structured_tools=False,
            token_logprobs=False,
            trainable=False,
            checkpoint_format=None,
        )

    def probe(self) -> ProbeResult:
        return ProbeResult(
            available=True,
            model_id=self.model_id,
            capabilities=self.capabilities,
            device="cpu",
            revision="deterministic-fixture-v1",
            reasons=("fixture caller; not realistic model-driven speech",),
        )

    def reset(self, seed: int) -> CallerTurn:
        self._seed = seed
        self._index = 0
        return self._render_next()

    def respond(self, agent_audio: bytes) -> CallerTurn | None:
        if self._index >= len(self.private_turns):
            return None
        return self._render_next()

    def _render_next(self) -> CallerTurn:
        index = self._index
        audio = self.renderer.render(self.private_turns[index], self.profile, self._seed + index)
        self._index += 1
        return CallerTurn(audio=audio, source="scripted_fixture", turn_index=index)
