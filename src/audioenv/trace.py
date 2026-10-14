"""Leakage-safe episode trace records."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from typing import Any

from .spec import JSONValue


@dataclass(frozen=True, slots=True)
class RewardEvent:
    name: str
    value: float
    category: str
    details: dict[str, JSONValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TraceStep:
    index: int
    action: dict[str, JSONValue]
    observation: dict[str, JSONValue]
    reward_events: tuple[RewardEvent, ...]
    state_hash: str
    latency_ms: float
    training: dict[str, JSONValue] = field(default_factory=dict)
    observability: dict[str, JSONValue] = field(default_factory=dict)


@dataclass(slots=True)
class EpisodeTrace:
    episode_id: str
    environment_id: str
    environment_version: str
    task_id: str
    seed: int
    model_id: str
    audio_profile_id: str
    steps: list[TraceStep] = field(default_factory=list)
    final_metrics: dict[str, JSONValue] = field(default_factory=dict)

    def to_dict(self, *, include_observability: bool = False) -> dict[str, Any]:
        data = asdict(self)
        if not include_observability:
            for step in data["steps"]:
                step["observability"] = {}
        return data

    def to_json(self, *, include_observability: bool = False) -> str:
        return json.dumps(self.to_dict(include_observability=include_observability), sort_keys=True, indent=2)


def state_hash(state: dict[str, JSONValue]) -> str:
    canonical = json.dumps(state, sort_keys=True, separators=(",", ":"))
    return sha256(canonical.encode()).hexdigest()

