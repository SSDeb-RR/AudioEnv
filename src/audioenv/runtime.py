"""State-isolated, turn-based audio environment runtime."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import time
from typing import Any
from uuid import uuid4

from .adapter import EnvironmentPackage, ToolResult
from .audio import AudioRenderer, DeterministicWavFixtureRenderer
from .rewards import RewardTracker, total_reward
from .spec import Action, AudioObservation, AudioProfile, JSONValue, SpeakAction, ToolCallAction
from .trace import EpisodeTrace, RewardEvent, TraceStep, state_hash


class EpisodeNotStarted(RuntimeError):
    pass


class AudioEnvironment:
    def __init__(self, package: EnvironmentPackage, renderer: AudioRenderer | None = None):
        self.package = package
        self.renderer = renderer or DeterministicWavFixtureRenderer()
        self._state: dict[str, JSONValue] | None = None
        self._task = None
        self._profile: AudioProfile | None = None
        self._seed = 0
        self._step_count = 0
        self._caller_index = 0
        self._done = False
        self._tracker: RewardTracker | None = None
        self._trace: EpisodeTrace | None = None

    def reset(
        self,
        task_id: str,
        seed: int = 0,
        *,
        audio_profile_id: str | None = None,
        model_id: str = "unknown",
    ) -> AudioObservation:
        task = self.package.task(task_id)
        profile = self._select_profile(audio_profile_id)
        self._task = task
        self._profile = profile
        self._seed = seed
        self._state = deepcopy(task.initial_state)
        self._step_count = 0
        self._caller_index = 0
        self._done = False
        self._tracker = RewardTracker(task)
        self._trace = EpisodeTrace(
            episode_id=str(uuid4()),
            environment_id=self.package.spec.id,
            environment_version=self.package.spec.version,
            task_id=task.id,
            seed=seed,
            model_id=model_id,
            audio_profile_id=profile.id,
        )
        audio = self.renderer.render(task.initial_utterance, profile, seed)
        return AudioObservation(audio=audio, metrics={"step_count": 0, "audio_profile_id": profile.id})

    def step(self, action: Action, *, training: dict[str, JSONValue] | None = None) -> AudioObservation:
        self._require_active()
        if self._done:
            raise RuntimeError("episode is already complete")
        assert self._task is not None and self._tracker is not None and self._state is not None
        started = time.perf_counter()
        self._step_count += 1
        events: tuple[RewardEvent, ...]
        audio: bytes | None = None
        tool_result: dict[str, JSONValue] | None = None
        observability: dict[str, JSONValue] = {}

        if isinstance(action, ToolCallAction):
            forbidden = action.name in self._task.forbidden_tools
            result = self._execute_tool(action, forbidden=forbidden)
            tool_result = result.observation
            events = self._tracker.record_tool(action, success=result.success, forbidden=forbidden)
            public_action: dict[str, JSONValue] = {
                "kind": "tool_call",
                "name": action.name,
                "arguments": action.arguments,
            }
        elif isinstance(action, SpeakAction):
            self._tracker.record_speech(action.semantic_text)
            observability = {"semantic_text": action.semantic_text}
            public_action = {
                "kind": "speak",
                "audio_sha256": state_hash({"audio": action.audio.hex()}),
                "end_episode": action.end_episode,
            }
            events = ()
            if self._caller_index < len(self._task.caller_turns) and not action.end_episode:
                private_turn = self._task.caller_turns[self._caller_index]
                self._caller_index += 1
                assert self._profile is not None
                audio = self.renderer.render(private_turn, self._profile, self._seed + self._caller_index)
            elif action.end_episode or self._caller_index >= len(self._task.caller_turns):
                self._done = True
        else:
            raise TypeError(f"unsupported action: {type(action)!r}")

        if self._step_count >= self.package.spec.max_steps:
            self._done = True
        terminal_metrics: dict[str, JSONValue] = {}
        if self._done:
            terminal, terminal_metrics = self._tracker.terminal(self._state)
            events = (*events, terminal)
        reward = total_reward(events)
        metrics: dict[str, JSONValue] = {"step_count": self._step_count, **terminal_metrics}
        observation = AudioObservation(audio=audio, tool_result=tool_result, reward=reward, done=self._done, metrics=metrics)
        assert self._trace is not None
        self._trace.steps.append(
            TraceStep(
                index=self._step_count,
                action=public_action,
                observation={
                    "has_audio": audio is not None,
                    "tool_result": tool_result,
                    "reward": reward,
                    "done": self._done,
                    "metrics": metrics,
                },
                reward_events=events,
                state_hash=state_hash(self._state),
                latency_ms=(time.perf_counter() - started) * 1000,
                training=training or {},
                observability=observability,
            )
        )
        if self._done:
            self._trace.final_metrics = metrics
        return observation

    @property
    def trace(self) -> EpisodeTrace:
        self._require_active()
        assert self._trace is not None
        return self._trace

    @property
    def public_state(self) -> dict[str, JSONValue]:
        self._require_active()
        return {
            "episode_id": self.trace.episode_id,
            "step_count": self._step_count,
            "done": self._done,
            "state_hash": state_hash(self._state or {}),
        }

    @property
    def state_snapshot(self) -> dict[str, JSONValue]:
        """Evaluator-facing snapshot. Do not place this in an agent context."""
        self._require_active()
        return deepcopy(self._state or {})

    def _execute_tool(self, action: ToolCallAction, *, forbidden: bool) -> ToolResult:
        if forbidden:
            return ToolResult(False, {"error": "unauthorized"})
        tool_spec = next((tool for tool in self.package.spec.tools if tool.name == action.name), None)
        if tool_spec is None or action.name not in self.package.tool_handlers:
            return ToolResult(False, {"error": "unknown_tool"})
        missing = [key for key in tool_spec.required_arguments if key not in action.arguments]
        if missing:
            return ToolResult(False, {"error": "missing_arguments", "arguments": missing})
        assert self._state is not None
        try:
            return self.package.tool_handlers[action.name](self._state, action.arguments)
        except (KeyError, TypeError, ValueError):
            return ToolResult(False, {"error": "tool_execution_failed"})

    def _select_profile(self, profile_id: str | None) -> AudioProfile:
        profiles = self.package.spec.audio_profiles
        if not profiles:
            raise ValueError("environment has no audio profiles")
        if profile_id is None:
            return profiles[0]
        for profile in profiles:
            if profile.id == profile_id:
                return profile
        raise KeyError(f"unknown audio profile: {profile_id}")

    def _require_active(self) -> None:
        if self._task is None:
            raise EpisodeNotStarted("call reset() before accessing the episode")

