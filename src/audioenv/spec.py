"""Canonical, dependency-free environment contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
from typing import Any, Literal, TypeAlias


JSONValue: TypeAlias = str | int | float | bool | None | list["JSONValue"] | dict[str, "JSONValue"]


@dataclass(frozen=True, slots=True)
class AudioProfile:
    id: str = "clean-en"
    language: str = "en"
    speaker: str = "synthetic-reference"
    sample_rate_hz: int = 16_000
    noise_snr_db: float | None = None
    codec: str = "pcm_s16le"
    speaking_rate: float = 1.0
    accent: str | None = None
    prosody: str = "neutral"


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    required_arguments: tuple[str, ...] = ()
    mutates_state: bool = False


@dataclass(frozen=True, slots=True)
class ToolExpectation:
    name: str
    arguments: dict[str, JSONValue]
    milestone: str
    reward: float = 0.1


@dataclass(frozen=True, slots=True)
class ToolCallAction:
    name: str
    arguments: dict[str, JSONValue]
    kind: Literal["tool_call"] = "tool_call"


@dataclass(frozen=True, slots=True)
class SpeakAction:
    audio: bytes
    semantic_text: str | None = None
    end_episode: bool = False
    kind: Literal["speak"] = "speak"


@dataclass(frozen=True, slots=True)
class InvalidAction:
    raw_sha256: str
    reason: str
    kind: Literal["invalid"] = "invalid"


Action: TypeAlias = ToolCallAction | SpeakAction | InvalidAction


@dataclass(frozen=True, slots=True)
class TaskSpec:
    id: str
    split: Literal["train", "validation", "test"]
    user_goal: str
    hidden_facts: dict[str, JSONValue]
    initial_state: dict[str, JSONValue]
    expected_state: dict[str, JSONValue]
    initial_utterance: str
    caller_turns: tuple[str, ...] = ()
    required_tools: tuple[ToolExpectation, ...] = ()
    required_communications: tuple[str, ...] = ()
    forbidden_tools: tuple[str, ...] = ()
    oracle_actions: tuple[ToolCallAction | SpeakAction, ...] = field(default=(), repr=False)


@dataclass(frozen=True, slots=True)
class EnvironmentSpec:
    id: str
    version: str
    description: str
    policies: tuple[str, ...]
    tools: tuple[ToolSpec, ...]
    audio_profiles: tuple[AudioProfile, ...]
    max_steps: int = 50
    model_requirements: tuple[str, ...] = ("audio-input", "audio-output", "structured-tool-calls")


@dataclass(frozen=True, slots=True)
class AudioObservation:
    audio: bytes | None = None
    tool_result: dict[str, JSONValue] | None = None
    reward: float = 0.0
    done: bool = False
    metrics: dict[str, JSONValue] = field(default_factory=dict)

    def agent_view(self) -> dict[str, JSONValue]:
        """Return only information allowed to enter the agent context."""
        return {
            "audio_hex": self.audio.hex() if self.audio is not None else None,
            "tool_result": self.tool_result,
            "reward": self.reward,
            "done": self.done,
            "metrics": self.metrics,
        }


def public_dataclass(value: Any) -> dict[str, Any]:
    """Serialize a public contract while excluding authoring-only oracle actions."""
    data = asdict(value)
    data.pop("oracle_actions", None)
    return _json_safe(data)


def _json_safe(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"sha256": sha256(value).hexdigest(), "size_bytes": len(value)}
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value

