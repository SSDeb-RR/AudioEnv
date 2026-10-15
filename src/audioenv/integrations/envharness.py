"""EnvHarness-compatible bridge and safe declarative adaptation layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..adapter import EnvironmentPackage
from ..runtime import AudioEnvironment
from ..spec import Action, SpeakAction, ToolCallAction


@dataclass(frozen=True, slots=True)
class BridgeResponse:
    observation: dict[str, Any]
    reward: float
    done: bool


class AudioEnvActionableBridge:
    """Duck-types EnvHarness's seven-method ActionableEnv boundary.

    Official type objects can be added by an optional shim without making the
    AudioEnv core depend on EnvHarness or LiteLLM.
    """

    def __init__(self, package: EnvironmentPackage, task_id: str | None = None):
        self.package = package
        self.task_id = task_id or package.tasks[0].id
        self.runtime = AudioEnvironment(package)
        self._last = None

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> BridgeResponse:
        options = options or {}
        task_id = str(options.get("task_id", self.task_id))
        self._last = self.runtime.reset(task_id, seed or 0, model_id="envharness-bridge")
        return self._response()

    def step(self, action: Action | dict[str, Any]) -> BridgeResponse:
        if isinstance(action, dict):
            kind = action.get("kind")
            if kind == "tool_call":
                action = ToolCallAction(str(action["name"]), dict(action.get("arguments", {})))
            elif kind == "speak":
                action = SpeakAction(
                    bytes.fromhex(str(action.get("audio_hex", ""))),
                    action.get("semantic_text"),
                    bool(action.get("end_episode")),
                )
            else:
                raise ValueError(f"unsupported bridge action kind: {kind}")
        self._last = self.runtime.step(action)
        return self._response()

    def observe(self) -> dict[str, Any]:
        if self._last is None:
            raise RuntimeError("reset the bridge before observe")
        return self._last.agent_view()

    def evaluate(self) -> dict[str, Any]:
        return dict(self.runtime.trace.final_metrics)

    def get_env_state(self) -> dict[str, Any]:
        return self.runtime.state_snapshot

    def save_state(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "public_state": self.runtime.public_state,
            "state": self.runtime.state_snapshot,
            "trace": self.runtime.trace.to_dict(include_observability=False),
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]):
        raise NotImplementedError(
            "Restoring a live runtime requires private tracker state; V1 refuses partial restore"
        )

    def _response(self) -> BridgeResponse:
        assert self._last is not None
        return BridgeResponse(self._last.agent_view(), self._last.reward, self._last.done)


class SetupAdapter:
    """Replay a fixed, validated action prefix after each reset."""

    def __init__(self, inner: AudioEnvActionableBridge, actions: tuple[Action, ...]):
        self.inner = inner
        self.actions = actions

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> BridgeResponse:
        response = self.inner.reset(seed, options)
        for action in self.actions:
            if response.done:
                raise RuntimeError("setup actions terminated the episode")
            response = self.inner.step(action)
        return response

    def __getattr__(self, name: str):
        return getattr(self.inner, name)


class RestrictedRulesAdapter:
    """Restrict allowed tools without changing the trusted evaluator."""

    def __init__(self, inner: AudioEnvActionableBridge, allowed_tools: frozenset[str]):
        self.inner = inner
        self.allowed_tools = allowed_tools

    def step(self, action: Action | dict[str, Any]) -> BridgeResponse:
        name = action.get("name") if isinstance(action, dict) else getattr(action, "name", None)
        if name is not None and name not in self.allowed_tools:
            raise ValueError(f"tool blocked by declarative rule: {name}")
        return self.inner.step(action)

    def __getattr__(self, name: str):
        return getattr(self.inner, name)


try:  # Optional exact integration against the pinned official package.
    from envharness.core.actionable_env import ActionableEnv as _OfficialActionableEnv
    from envharness.core.types import (
        Action as _OfficialAction,
        EnvResetResponse as _OfficialReset,
        EnvResponse as _OfficialResponse,
        EvaluationResult as _OfficialEvaluation,
        Observation as _OfficialObservation,
    )
except ImportError:  # pragma: no cover - normal core-only installation
    OfficialAudioEnvBridge = None
else:
    _PACKAGES: dict[str, EnvironmentPackage] = {}

    class OfficialAudioEnvBridge(_OfficialActionableEnv):
        """Concrete implementation of the installed EnvHarness interface."""

        tool_registry = []

        def __init__(self, package: EnvironmentPackage, task_id: str | None = None):
            self.package = package
            self.task_id = task_id or package.tasks[0].id
            self.runtime = AudioEnvironment(package)
            self._last = None
            self._seed: int | None = None
            self._options: dict[str, Any] = {}
            _PACKAGES[package.spec.id] = package

        @classmethod
        def env_type(cls) -> str:
            return "audioenv-v1"

        def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
            self._seed, self._options = seed, options or {}
            task_id = str(self._options.get("task_id", self.task_id))
            self._last = self.runtime.reset(task_id, seed or 0, model_id="envharness-official")
            return _OfficialReset(observation=self.observe(), info={"task_id": task_id})

        def step(self, action):
            if action.name == "speak":
                internal = SpeakAction(
                    bytes.fromhex(str(action.kwargs.get("audio_hex", ""))),
                    action.kwargs.get("semantic_text"),
                    bool(action.kwargs.get("end_episode")),
                )
            else:
                internal = ToolCallAction(action.name, dict(action.kwargs))
            self._last = self.runtime.step(internal)
            return _OfficialResponse(
                observation=self.observe(), reward=self._last.reward,
                terminated=self._last.done, truncated=False,
                info={"audioenv_metrics": self._last.metrics},
            )

        def observe(self):
            if self._last is None:
                raise RuntimeError("reset before observe")
            return _OfficialObservation(text="", data=self._last.agent_view())

        def evaluate(self):
            metrics = dict(self.runtime.trace.final_metrics)
            success = bool(metrics.get("terminal_pass"))
            return _OfficialEvaluation(success=success, score=float(success), metrics=metrics)

        def get_env_state(self):
            return self.runtime.state_snapshot

        def save_state(self):
            return {
                "package_id": self.package.spec.id,
                "task_id": self.task_id,
                "seed": self._seed,
                "options": self._options,
            }

        @classmethod
        def from_state(cls, state: dict):
            package_id = str(state["package_id"])
            if package_id not in _PACKAGES:
                raise ValueError("package must be registered in this process before restore")
            return cls(_PACKAGES[package_id], str(state["task_id"]))

        def default_reset_args(self):
            return self._seed, self._options

