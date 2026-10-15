"""Benchmark adapter SDK and executable package type."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from .spec import Action, EnvironmentSpec, JSONValue, TaskSpec


@dataclass(frozen=True, slots=True)
class ToolResult:
    success: bool
    observation: dict[str, JSONValue]


ToolHandler = Callable[[dict[str, JSONValue], dict[str, JSONValue]], ToolResult]


@dataclass(frozen=True, slots=True)
class EnvironmentPackage:
    spec: EnvironmentSpec
    tasks: tuple[TaskSpec, ...]
    tool_handlers: dict[str, ToolHandler]
    public_action_candidates: dict[str, tuple[Action, ...]] = field(default_factory=dict)

    def task(self, task_id: str) -> TaskSpec:
        for task in self.tasks:
            if task.id == task_id:
                return task
        raise KeyError(f"unknown task: {task_id}")


class BenchmarkAdapter(Protocol):
    """Implement this interface without modifying the runtime core."""

    def inspect(self, source: Path) -> dict[str, Any]: ...

    def convert(self, source: Path) -> EnvironmentPackage: ...

