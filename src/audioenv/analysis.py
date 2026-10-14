"""Deterministic, actionable failure classification."""

from __future__ import annotations

from .trace import EpisodeTrace


def classify_failures(trace: EpisodeTrace) -> tuple[str, ...]:
    failures: set[str] = set()
    for step in trace.steps:
        for event in step.reward_events:
            if event.name == "unauthorized_mutation":
                failures.add("incorrect_authorization")
            elif event.name == "repeated_failed_call":
                failures.add("dead_loop")
            elif event.name == "failed_tool_call":
                failures.add("wrong_tool_or_arguments")
    metrics = trace.final_metrics
    if metrics and not metrics.get("communication_ok", True):
        failures.add("incomplete_communication")
    if metrics and not metrics.get("state_ok", True):
        failures.add("incorrect_final_state")
    if metrics and not metrics.get("tools_ok", True):
        failures.add("missing_required_action")
    return tuple(sorted(failures))

