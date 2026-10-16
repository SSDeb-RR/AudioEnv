"""Protected tau2 conversation state machine for native-audio rollouts.

This module deliberately knows nothing about VeRL or Qwen tensors.  A policy
port receives only caller audio and permitted tool results.  The private tau2
scenario is sent directly to the frozen caller service and never returned in
the episode result or policy observation.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
from typing import Any, Awaitable, Callable, Protocol


class Tau2Port(Protocol):
    def reset(self, task_id: str) -> dict[str, Any]: ...
    def caller_context(self, session_id: str) -> dict[str, Any]: ...
    def speak(self, session_id: str, role: str, semantic_text: str, *, end_episode: bool = False) -> dict[str, Any]: ...
    def tool(self, session_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]: ...
    def verify(self, session_id: str) -> dict[str, Any]: ...


class CallerPort(Protocol):
    def create(self, private_goal: str, seed: int) -> dict[str, Any]: ...
    def respond(self, session_id: str, agent_audio: bytes) -> dict[str, Any]: ...
    def delete(self, session_id: str) -> None: ...


@dataclass(frozen=True, slots=True)
class PolicyObservation:
    caller_audio: bytes
    tool_result: dict[str, Any] | None
    turn_index: int
    tool_schemas: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class PolicyTurn:
    semantic_text: str
    audio: bytes
    metadata: Any


@dataclass(frozen=True, slots=True)
class EpisodeAction:
    kind: str
    semantic_text: str = ""
    tool_name: str | None = None
    tool_arguments: dict[str, Any] | None = None
    terminate: bool = False


@dataclass(frozen=True, slots=True)
class EpisodeStep:
    index: int
    action: EpisodeAction
    policy_metadata: Any
    tool_result: dict[str, Any] | None


@dataclass(frozen=True, slots=True)
class Tau2EpisodeResult:
    task_id: str
    steps: tuple[EpisodeStep, ...]
    verification: dict[str, Any]
    terminal_reward: float
    process_reward: float
    truncated: bool
    truncation_reason: str | None


PolicyGenerate = Callable[[PolicyObservation], PolicyTurn | Awaitable[PolicyTurn]]


class Tau2NativeEpisodeController:
    """Run one protected multi-turn episode through explicit service ports."""

    def __init__(self, tau2: Tau2Port, caller: CallerPort, *, max_turns: int = 16):
        if max_turns < 1:
            raise ValueError("max_turns must be positive")
        self.tau2, self.caller, self.max_turns = tau2, caller, max_turns

    async def run(self, task_id: str, seed: int, generate: PolicyGenerate) -> Tau2EpisodeResult:
        reset = await asyncio.to_thread(self.tau2.reset, task_id)
        tau2_session = _required_string(reset, "session_id")
        raw_tools = reset.get("tool_schemas") or ()
        if not isinstance(raw_tools, (list, tuple)) or any(not isinstance(item, dict) for item in raw_tools):
            raise ValueError("tau2 reset returned invalid tool schemas")
        tool_schemas = tuple(raw_tools)
        private = await asyncio.to_thread(self.tau2.caller_context, tau2_session)
        # The complete object is serialized only across the caller-only port.
        # No copy is retained in the result or passed to ``generate``.
        opened = await asyncio.to_thread(
            self.caller.create, json.dumps(private, sort_keys=True, separators=(",", ":")), seed,
        )
        caller_session = _required_string(opened, "session_id")
        caller_audio = _audio(opened)
        caller_text = str(opened.get("semantic_text") or "").strip()
        if caller_text:
            await asyncio.to_thread(self.tau2.speak, tau2_session, "user", caller_text)

        steps: list[EpisodeStep] = []
        tool_result = None
        terminated = False
        reason = None
        process_reward = 0.0
        failed_actions: set[str] = set()
        try:
            for index in range(1, self.max_turns + 1):
                produced = generate(PolicyObservation(caller_audio, tool_result, index, tool_schemas))
                turn = await produced if hasattr(produced, "__await__") else produced
                if not isinstance(turn, PolicyTurn):
                    raise TypeError("policy generator must return PolicyTurn")
                try:
                    action = parse_episode_action(turn.semantic_text)
                except ValueError:
                    action = EpisodeAction(kind="invalid", semantic_text=turn.semantic_text)
                    tool_result = {"result": {"error": "invalid_action"}}
                    process_reward += _failure_penalty(
                        "invalid:" + turn.semantic_text, failed_actions,
                    )
                if action.kind == "tool":
                    try:
                        raw_tool_result = await asyncio.to_thread(
                            self.tau2.tool,
                            tau2_session, action.tool_name or "", action.tool_arguments or {},
                        )
                        # Worker diagnostics such as the database hash are useful
                        # for traces but are not a tool observation for the model.
                        tool_result = _public_tool_result(raw_tool_result)
                    except Exception:
                        # Do not terminate an expensive rollout or expose worker
                        # internals. The policy receives one stable recovery signal.
                        tool_result = {"result": {"error": "tool_execution_failed"}}
                        signature = json.dumps(
                            {"name": action.tool_name, "arguments": action.tool_arguments},
                            sort_keys=True, separators=(",", ":"),
                        )
                        process_reward += _failure_penalty("tool:" + signature, failed_actions)
                elif action.kind == "speak":
                    if not turn.audio:
                        raise ValueError("spoken policy action requires bound native audio")
                    await asyncio.to_thread(
                        self.tau2.speak, tau2_session, "assistant", action.semantic_text,
                        end_episode=action.terminate,
                    )
                    tool_result = None
                    if action.terminate:
                        terminated = True
                    else:
                        response = await asyncio.to_thread(self.caller.respond, caller_session, turn.audio)
                        caller_audio = _audio(response)
                        caller_text = str(response.get("semantic_text") or "").strip()
                        if caller_text:
                            await asyncio.to_thread(self.tau2.speak, tau2_session, "user", caller_text)
                elif action.kind != "invalid":
                    raise AssertionError("unreachable action kind")
                steps.append(EpisodeStep(index, action, turn.metadata, tool_result))
                if terminated:
                    break
            else:
                reason = "max_turns"
            verification = await asyncio.to_thread(self.tau2.verify, tau2_session)
        finally:
            await asyncio.to_thread(self.caller.delete, caller_session)
        return Tau2EpisodeResult(
            task_id=task_id, steps=tuple(steps), verification=verification,
            terminal_reward=float(verification.get("upstream_reward", 0.0)),
            process_reward=process_reward,
            truncated=not terminated, truncation_reason=reason,
        )


def parse_episode_action(text: str) -> EpisodeAction:
    """Parse the one supported structured envelope or a normal spoken turn."""

    stripped = text.strip()
    if not stripped:
        raise ValueError("policy emitted an empty action")
    if not stripped.startswith("{"):
        return EpisodeAction(kind="speak", semantic_text=stripped)
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError("malformed structured action envelope") from exc
    if not isinstance(payload, dict):
        raise ValueError("structured action envelope must be an object")
    if set(payload) == {"tool_call"} and isinstance(payload["tool_call"], dict):
        call = payload["tool_call"]
        if set(call) != {"name", "arguments"} or not isinstance(call["name"], str) or not isinstance(call["arguments"], dict):
            raise ValueError("tool_call requires exactly name and arguments")
        return EpisodeAction(kind="tool", tool_name=call["name"], tool_arguments=call["arguments"])
    if set(payload) == {"speak", "terminate"} and isinstance(payload["speak"], str) and isinstance(payload["terminate"], bool):
        spoken = payload["speak"].strip()
        if not spoken:
            raise ValueError("speak text cannot be empty")
        return EpisodeAction(kind="speak", semantic_text=spoken, terminate=payload["terminate"])
    raise ValueError("unsupported structured action envelope")


def _required_string(payload: dict[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"missing {field}")
    return value


def _audio(payload: dict[str, Any]) -> bytes:
    import base64

    encoded = payload.get("audio_base64")
    if not isinstance(encoded, str) or not encoded:
        raise ValueError("caller response is missing audio")
    try:
        decoded = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise ValueError("caller response contains invalid audio") from exc
    if not decoded:
        raise ValueError("caller response contains empty audio")
    return decoded


def _public_tool_result(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict) or "result" not in payload:
        raise ValueError("tau2 tool response is missing its public result")
    return {"result": payload["result"]}


def _failure_penalty(signature: str, previous: set[str]) -> float:
    """Penalize a failed action and penalize an identical retry once more."""

    repeated = signature in previous
    previous.add(signature)
    return -0.2 if repeated else -0.1
