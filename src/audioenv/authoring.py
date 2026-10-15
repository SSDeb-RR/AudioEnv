"""Validated, declarative custom-environment authoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .adapter import EnvironmentPackage, ToolResult
from .artifacts import LocalArtifactStore
from .contracts import (
    AuthoringStatus,
    EnvironmentPackageV1,
    EnvironmentType,
    ProtectedVerifierRef,
    PublicTaskV1,
    SourceProvenance,
)
from .spec import AudioProfile, EnvironmentSpec, SpeakAction, TaskSpec, ToolCallAction, ToolExpectation, ToolSpec
from .validation import ValidationReport, validate_package


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DeclarativeTool(StrictModel):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    description: str
    operation: Literal["read_path", "set_path"]
    path: tuple[str, ...]
    required_arguments: tuple[str, ...] = ()
    value_argument: str | None = None

    @model_validator(mode="after")
    def check_operation(self) -> "DeclarativeTool":
        if not self.path:
            raise ValueError("tool path cannot be empty")
        if self.operation == "set_path" and not self.value_argument:
            raise ValueError("set_path requires value_argument")
        if self.value_argument and self.value_argument not in self.required_arguments:
            raise ValueError("value_argument must be a required argument")
        return self


class CustomTaskInput(StrictModel):
    environment_id: str
    version: str = "1.0.0"
    description: str
    policies: tuple[str, ...] = ()
    task_id: str
    split: Literal["train", "validation", "test"] = "train"
    goal: str
    initial_state: dict[str, Any]
    expected_state: dict[str, Any]
    initial_utterance: str
    caller_turns: tuple[str, ...] = ()
    tools: tuple[DeclarativeTool, ...]
    required_tools: tuple[dict[str, Any], ...]
    required_communications: tuple[str, ...] = ()
    forbidden_tools: tuple[str, ...] = ()
    max_steps: int = Field(default=20, ge=1, le=500)


@dataclass(frozen=True, slots=True)
class AuthoredEnvironment:
    package: EnvironmentPackage
    boundary: EnvironmentPackageV1
    validation: ValidationReport


class DeterministicEnvironmentAuthor:
    """Compile a constrained task specification without executing generated code."""

    def __init__(self, artifacts: LocalArtifactStore):
        self.artifacts = artifacts

    def author(self, source: CustomTaskInput) -> AuthoredEnvironment:
        tool_specs = tuple(
            ToolSpec(
                item.name,
                item.description,
                item.required_arguments,
                item.operation == "set_path",
            )
            for item in source.tools
        )
        handlers = {item.name: _handler(item) for item in source.tools}
        expectations = tuple(
            ToolExpectation(
                name=str(item["name"]),
                arguments=dict(item.get("arguments", {})),
                milestone=str(item.get("milestone", item["name"])),
                reward=float(item.get("reward", 0.1)),
            )
            for item in source.required_tools
        )
        oracle_actions = tuple(
            ToolCallAction(item.name, dict(item.arguments)) for item in expectations
        ) + (SpeakAction(b"authored-oracle", " ".join(source.required_communications), True),)
        task = TaskSpec(
            id=source.task_id,
            split=source.split,
            user_goal=source.goal,
            hidden_facts={},
            initial_state=source.initial_state,
            expected_state=source.expected_state,
            initial_utterance=source.initial_utterance,
            caller_turns=source.caller_turns,
            required_tools=expectations,
            required_communications=source.required_communications,
            forbidden_tools=source.forbidden_tools,
            oracle_actions=oracle_actions,
        )
        spec = EnvironmentSpec(
            id=source.environment_id,
            version=source.version,
            description=source.description,
            policies=source.policies,
            tools=tool_specs,
            audio_profiles=(AudioProfile(),),
            max_steps=source.max_steps,
        )
        package = EnvironmentPackage(spec, (task,), handlers)
        validation = validate_package(package)
        verifier_payload = {
            "task_id": task.id,
            "expected_state": task.expected_state,
            "required_tools": [
                {"name": item.name, "arguments": item.arguments, "milestone": item.milestone}
                for item in expectations
            ],
            "required_communications": task.required_communications,
            "forbidden_tools": task.forbidden_tools,
        }
        verifier = self.artifacts.put_json(verifier_payload)
        payload = {
            "schema_version": "audioenv.environment/v1",
            "id": spec.id,
            "version": spec.version,
            "description": spec.description,
            "environment_type": EnvironmentType.CONVERSATIONAL,
            "source": SourceProvenance(kind="custom").model_dump(mode="json"),
            "required_capabilities": spec.model_requirements,
            "tasks": [
                PublicTaskV1(
                    id=task.id,
                    split=task.split,
                    goal=task.user_goal,
                    tool_names=tuple(item.name for item in spec.tools),
                    audio_profile_ids=tuple(item.id for item in spec.audio_profiles),
                ).model_dump(mode="json")
            ],
            "protected_verifier": ProtectedVerifierRef(
                artifact_id=verifier.id, sha256=verifier.sha256
            ).model_dump(mode="json"),
            "authoring_status": AuthoringStatus.VALIDATED if validation.passed else AuthoringStatus.MANUAL,
            "parent_hash": None,
        }
        payload["content_hash"] = EnvironmentPackageV1.hash_payload(payload)
        boundary = EnvironmentPackageV1.model_validate(payload)
        return AuthoredEnvironment(package, boundary, validation)


def _handler(spec: DeclarativeTool):
    def execute(state: dict[str, Any], arguments: dict[str, Any]) -> ToolResult:
        cursor: Any = state
        for key in spec.path[:-1]:
            if not isinstance(cursor, dict) or key not in cursor:
                return ToolResult(False, {"error": "path_not_found"})
            cursor = cursor[key]
        if not isinstance(cursor, dict):
            return ToolResult(False, {"error": "path_not_found"})
        leaf = spec.path[-1]
        if spec.operation == "read_path":
            if leaf not in cursor:
                return ToolResult(False, {"error": "path_not_found"})
            return ToolResult(True, {"value": cursor[leaf]})
        assert spec.value_argument is not None
        cursor[leaf] = arguments[spec.value_argument]
        return ToolResult(True, {"value": cursor[leaf]})

    return execute

