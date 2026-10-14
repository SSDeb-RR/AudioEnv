"""Adapter for a portable structured text-agent benchmark manifest."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from ..adapter import EnvironmentPackage, ToolHandler
from ..spec import AudioProfile, EnvironmentSpec, SpeakAction, TaskSpec, ToolCallAction, ToolExpectation, ToolSpec


class JsonBenchmarkAdapter:
    """Convert JSON benchmark truth while binding executable tools separately.

    Keeping handlers out of data files prevents the importer from executing
    untrusted code. Callers explicitly supply the audited handler registry.
    """

    def __init__(self, tool_handlers: dict[str, ToolHandler]):
        self.tool_handlers = tool_handlers

    def inspect(self, source: Path) -> dict[str, Any]:
        payload = self._load(source)
        return {
            "environment_id": payload["environment"]["id"],
            "task_count": len(payload.get("tasks", [])),
            "tool_count": len(payload["environment"].get("tools", [])),
            "splits": sorted({task["split"] for task in payload.get("tasks", [])}),
        }

    def convert(self, source: Path) -> EnvironmentPackage:
        payload = self._load(source)
        raw_spec = payload["environment"]
        tools = tuple(
            ToolSpec(
                name=tool["name"],
                description=tool.get("description", ""),
                required_arguments=tuple(tool.get("required_arguments", [])),
                mutates_state=bool(tool.get("mutates_state", False)),
            )
            for tool in raw_spec["tools"]
        )
        missing = {tool.name for tool in tools} - set(self.tool_handlers)
        if missing:
            raise ValueError(f"missing audited tool handlers: {sorted(missing)}")
        profiles = tuple(AudioProfile(**profile) for profile in raw_spec["audio_profiles"])
        spec = EnvironmentSpec(
            id=raw_spec["id"],
            version=raw_spec["version"],
            description=raw_spec.get("description", ""),
            policies=tuple(raw_spec.get("policies", [])),
            tools=tools,
            audio_profiles=profiles,
            max_steps=int(raw_spec.get("max_steps", 50)),
            model_requirements=tuple(
                raw_spec.get("model_requirements", ("audio-input", "audio-output", "structured-tool-calls"))
            ),
        )
        tasks = tuple(self._task(task) for task in payload["tasks"])
        return EnvironmentPackage(spec, tasks, {tool.name: self.tool_handlers[tool.name] for tool in tools})

    @staticmethod
    def _load(source: Path) -> dict[str, Any]:
        payload = json.loads(source.read_text(encoding="utf-8"))
        if payload.get("format") != "audioenv/v1":
            raise ValueError("expected format audioenv/v1")
        return payload

    @staticmethod
    def _action(raw: dict[str, Any]):
        if raw["kind"] == "tool_call":
            return ToolCallAction(raw["name"], raw.get("arguments", {}))
        if raw["kind"] == "speak":
            audio = base64.b64decode(raw.get("audio_base64", ""))
            return SpeakAction(audio, raw.get("semantic_text"), bool(raw.get("end_episode", False)))
        raise ValueError(f"unknown oracle action kind: {raw['kind']}")

    def _task(self, raw: dict[str, Any]) -> TaskSpec:
        return TaskSpec(
            id=raw["id"],
            split=raw["split"],
            user_goal=raw["user_goal"],
            hidden_facts=raw.get("hidden_facts", {}),
            initial_state=raw["initial_state"],
            expected_state=raw["expected_state"],
            initial_utterance=raw["initial_utterance"],
            caller_turns=tuple(raw.get("caller_turns", [])),
            required_tools=tuple(
                ToolExpectation(
                    expectation["name"],
                    expectation.get("arguments", {}),
                    expectation["milestone"],
                    float(expectation.get("reward", 0.1)),
                )
                for expectation in raw.get("required_tools", [])
            ),
            required_communications=tuple(raw.get("required_communications", [])),
            forbidden_tools=tuple(raw.get("forbidden_tools", [])),
            oracle_actions=tuple(self._action(action) for action in raw.get("oracle_actions", [])),
        )

