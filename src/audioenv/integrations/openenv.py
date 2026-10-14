"""OpenEnv server-side adapter.

The dependency is optional. When installed, ``make_openenv_environment``
returns a concrete OpenEnv Environment implementing reset/step/state.
"""

from __future__ import annotations

import base64
from typing import Any

from ..adapter import EnvironmentPackage
from ..runtime import AudioEnvironment
from ..spec import SpeakAction, ToolCallAction


def _parse_action(action: Any):
    kind = getattr(action, "kind", None) or action.get("kind")
    if kind == "tool_call":
        name = getattr(action, "name", None) or action["name"]
        arguments = getattr(action, "arguments", None) or action.get("arguments", {})
        return ToolCallAction(name, arguments)
    if kind == "speak":
        encoded = getattr(action, "audio_base64", None) or action.get("audio_base64", "")
        semantic = getattr(action, "semantic_text", None) if not isinstance(action, dict) else action.get("semantic_text")
        end = getattr(action, "end_episode", False) if not isinstance(action, dict) else action.get("end_episode", False)
        return SpeakAction(base64.b64decode(encoded), semantic, end)
    raise ValueError(f"unknown action kind: {kind}")


def make_openenv_environment(package: EnvironmentPackage, default_task_id: str):
    try:
        from openenv.core.env_server.interfaces import Environment
        from openenv.core.env_server.types import Observation, State
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Install with `pip install audioenv-foundry[openenv]`") from exc

    class AudioEnvObservation(Observation):
        audio_base64: str | None = None
        tool_result: dict[str, Any] | None = None
        metrics: dict[str, Any] = {}

    class AudioEnvState(State):
        task_id: str = default_task_id
        state_hash: str = ""

    class OpenEnvAudioEnvironment(Environment):
        def __init__(self):
            self.runtime = AudioEnvironment(package)
            self._state = AudioEnvState()

        def reset(self, seed=None, episode_id=None, **kwargs):
            task_id = kwargs.get("task_id", default_task_id)
            result = self.runtime.reset(task_id, seed or 0, model_id=kwargs.get("model_id", "openenv"))
            self._state = AudioEnvState(
                episode_id=self.runtime.trace.episode_id,
                step_count=0,
                task_id=task_id,
                state_hash=str(self.runtime.public_state["state_hash"]),
            )
            return AudioEnvObservation(
                audio_base64=base64.b64encode(result.audio or b"").decode(),
                reward=result.reward,
                done=result.done,
                metrics=result.metrics,
            )

        def step(self, action, timeout_s=None, **kwargs):
            result = self.runtime.step(_parse_action(action))
            self._state.step_count = int(self.runtime.public_state["step_count"])
            self._state.state_hash = str(self.runtime.public_state["state_hash"])
            return AudioEnvObservation(
                audio_base64=base64.b64encode(result.audio).decode() if result.audio else None,
                tool_result=result.tool_result,
                reward=result.reward,
                done=result.done,
                metrics=result.metrics,
            )

        @property
        def state(self):
            return self._state

    return OpenEnvAudioEnvironment

