"""Runtime hand-off used by the pinned VeRL AgentLoopBase integration.

The policy server is owned by VeRL, so AudioEnv's native runtime is created
inside the agent-loop worker.  A worker image must provide one reviewed runtime
factory via ``AUDIOENV_NATIVE_RUNTIME_FACTORY=module:attribute``.  This module
does not fall back to a scripted or oracle rollout.
"""

from __future__ import annotations

import importlib
import os
from typing import Any, Callable

from ..contracts import TrajectoryV2


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


async def run_audioenv_trajectory(*, agent_loop, sampling_params, **sample) -> TrajectoryV2:
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
    if not isinstance(result, TrajectoryV2):
        raise TypeError("native AudioEnv runtime must return TrajectoryV2")
    return result
