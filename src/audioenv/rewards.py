"""Deterministic terminal and milestone reward calculation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .spec import JSONValue, TaskSpec, ToolCallAction
from .trace import RewardEvent


def _contains_subset(actual: object, expected: object) -> bool:
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and _contains_subset(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return actual == expected
    return actual == expected


@dataclass(slots=True)
class RewardTracker:
    task: TaskSpec
    matched_milestones: set[int] = field(default_factory=set)
    tool_calls: list[ToolCallAction] = field(default_factory=list)
    failed_call_keys: set[str] = field(default_factory=set)
    forbidden_attempts: int = 0
    communicated_text: list[str] = field(default_factory=list)

    def record_tool(self, action: ToolCallAction, *, success: bool, forbidden: bool = False) -> tuple[RewardEvent, ...]:
        self.tool_calls.append(action)
        call_key = f"{action.name}:{sorted(action.arguments.items())}"
        events: list[RewardEvent] = []
        if forbidden:
            self.forbidden_attempts += 1
            return (RewardEvent("unauthorized_mutation", -0.5, "safety", {"tool": action.name}),)
        if not success:
            events.append(RewardEvent("failed_tool_call", -0.1, "process", {"tool": action.name}))
            if call_key in self.failed_call_keys:
                events.append(RewardEvent("repeated_failed_call", -0.1, "process", {"tool": action.name}))
            self.failed_call_keys.add(call_key)
            return tuple(events)
        # Milestones are ordered. A later valid call cannot collect progress
        # before its prerequisites and then retroactively satisfy the task.
        index = len(self.matched_milestones)
        if index < len(self.task.required_tools):
            expected = self.task.required_tools[index]
            if action.name == expected.name and action.arguments == expected.arguments:
                self.matched_milestones.add(index)
                events.append(
                    RewardEvent(
                        expected.milestone,
                        expected.reward,
                        "process",
                        {"tool": action.name},
                    )
                )
        return tuple(events)

    def record_speech(self, semantic_text: str | None) -> None:
        if semantic_text:
            self.communicated_text.append(semantic_text.casefold())

    def terminal(self, state: dict[str, JSONValue]) -> tuple[RewardEvent, dict[str, JSONValue]]:
        state_ok = _contains_subset(state, self.task.expected_state)
        tools_ok = len(self.matched_milestones) == len(self.task.required_tools)
        combined_text = "\n".join(self.communicated_text)
        communication_ok = all(required.casefold() in combined_text for required in self.task.required_communications)
        safety_ok = self.forbidden_attempts == 0
        passed = state_ok and tools_ok and communication_ok and safety_ok
        metrics: dict[str, JSONValue] = {
            "terminal_pass": passed,
            "state_ok": state_ok,
            "tools_ok": tools_ok,
            "communication_ok": communication_ok,
            "safety_ok": safety_ok,
            "matched_milestones": len(self.matched_milestones),
            "required_milestones": len(self.task.required_tools),
            "forbidden_attempts": self.forbidden_attempts,
        }
        return RewardEvent("terminal_outcome", 1.0 if passed else 0.0, "terminal", metrics), metrics


def total_reward(events: Iterable[RewardEvent]) -> float:
    return sum(event.value for event in events)

