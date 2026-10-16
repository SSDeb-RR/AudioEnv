"""Runtime hand-off used by the pinned VeRL AgentLoopBase integration.

The policy server is owned by VeRL, so AudioEnv's native runtime is created
inside the agent-loop worker.  A worker image must provide one reviewed runtime
factory via ``AUDIOENV_NATIVE_RUNTIME_FACTORY=module:attribute``.  This module
does not fall back to a scripted or oracle rollout.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass, field
from typing import Any, Callable

from ..contracts import TrajectoryV2


@dataclass(frozen=True, slots=True)
class NativeVerlRollout:
    """Unpadded native rollout handed to VeRL for multimodal postprocessing.

    Position IDs intentionally do not appear here. VeRL derives them with the
    loaded Qwen processor from these exact token and audio inputs.
    """

    prompt_ids: tuple[int, ...]
    response_ids: tuple[int, ...]
    response_mask: tuple[int, ...]
    response_logprobs: tuple[float, ...]
    # None forces VeRL's configured reward loop to invoke AudioEnv's
    # hash-bound scorer. Supplying a float would create rm_scores early and
    # silently bypass the custom reward function.
    reward_score: float | None
    num_turns: int
    audio_paths: tuple[str, ...]
    mm_processor_kwargs: dict[str, Any] = field(default_factory=dict)
    extra_fields: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.prompt_ids or not self.response_ids:
            raise ValueError("native VeRL rollout requires prompt and response tokens")
        if len(self.response_ids) != len(self.response_mask):
            raise ValueError("native rollout response mask is misaligned")
        if len(self.response_ids) != len(self.response_logprobs):
            raise ValueError("native rollout log-probabilities are misaligned")
        if not any(self.response_mask):
            raise ValueError("native rollout requires trainable policy tokens")
        if not self.audio_paths:
            raise ValueError("native Qwen rollout requires caller audio")


def _runtime_factory() -> Callable[..., Any]:
    reference = os.environ.get("AUDIOENV_NATIVE_RUNTIME_FACTORY", "")
    module_name, separator, attribute = reference.partition(":")
    if not separator or not module_name or not attribute:
        raise RuntimeError(
            "AUDIOENV_NATIVE_RUNTIME_FACTORY must identify the reviewed "
            "Qwen caller/environment runtime as module:attribute"
        )
    factory = getattr(importlib.import_module(module_name), attribute)
    if not callable(factory):
        raise TypeError("AUDIOENV native runtime factory is not callable")
    return factory


async def run_audioenv_trajectory(*, agent_loop, sampling_params, **sample) -> TrajectoryV2 | NativeVerlRollout:
    """Execute the configured native-audio runtime using VeRL's live policy.

    The factory receives the instantiated VeRL loop so it must generate policy
    turns through ``agent_loop.server_manager``. Returning any type other than
    the protected ``TrajectoryV2`` contract is rejected by the caller.
    """

    result = _runtime_factory()(
        agent_loop=agent_loop, sampling_params=sampling_params, sample=sample,
    )
    if hasattr(result, "__await__"):
        result = await result
    if not isinstance(result, (TrajectoryV2, NativeVerlRollout)):
        raise TypeError("native AudioEnv runtime must return TrajectoryV2 or NativeVerlRollout")
    return result
