"""Environment integrity checks used before evaluation or training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .adapter import EnvironmentPackage
from .audio import validate_audio_profile
from .runtime import AudioEnvironment
from .spec import SpeakAction, ToolCallAction


@dataclass(frozen=True, slots=True)
class CheckResult:
    name: str
    passed: bool
    details: str


@dataclass(frozen=True, slots=True)
class ValidationReport:
    checks: tuple[CheckResult, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)


def _finish(env: AudioEnvironment) -> None:
    if not env.public_state["done"]:
        env.step(SpeakAction(b"finish", "finished", True))


def validate_package(package: EnvironmentPackage) -> ValidationReport:
    checks: list[CheckResult] = []
    ids = [task.id for task in package.tasks]
    checks.append(CheckResult("unique_task_ids", len(ids) == len(set(ids)), f"{len(ids)} tasks"))
    tool_names = {tool.name for tool in package.spec.tools}
    handlers_ok = tool_names == set(package.tool_handlers)
    checks.append(CheckResult("tool_handlers_complete", handlers_ok, f"{len(tool_names)} declared tools"))
    for profile in package.spec.audio_profiles:
        try:
            validate_audio_profile(profile)
            profile_ok, details = True, "implemented audio transformations only"
        except ValueError as exc:
            profile_ok, details = False, str(exc)
        checks.append(CheckResult(f"audio_profile:{profile.id}", profile_ok, details))

    for task in package.tasks:
        known_tools = all(expectation.name in tool_names for expectation in task.required_tools)
        checks.append(CheckResult(f"{task.id}:known_tools", known_tools, "required tools are declared"))

        oracle = AudioEnvironment(package)
        initial = oracle.reset(task.id, seed=41, model_id="oracle")
        no_text_leak = "text" not in initial.agent_view() and task.initial_utterance.encode() not in (initial.audio or b"")
        checks.append(CheckResult(f"{task.id}:no_text_leak", no_text_leak, "reset exposes audio only"))
        for action in task.oracle_actions:
            oracle.step(action)
        _finish(oracle)
        oracle_passed = bool(oracle.trace.final_metrics.get("terminal_pass"))
        checks.append(CheckResult(f"{task.id}:oracle_pass", oracle_passed, "oracle terminal result"))

        noop = AudioEnvironment(package)
        noop.reset(task.id, seed=41, model_id="noop")
        noop.step(SpeakAction(b"noop", "I cannot help.", True))
        noop_failed = not bool(noop.trace.final_metrics.get("terminal_pass"))
        checks.append(CheckResult(f"{task.id}:noop_fail", noop_failed, "no-op rejected"))

        isolated_a = AudioEnvironment(package)
        isolated_b = AudioEnvironment(package)
        isolated_a.reset(task.id, seed=41, model_id="isolation-a")
        isolated_b.reset(task.id, seed=41, model_id="isolation-b")
        pristine_b = isolated_b.state_snapshot
        if task.oracle_actions:
            isolated_a.step(task.oracle_actions[0])
        isolated = isolated_b.state_snapshot == pristine_b and isolated_a.public_state is not isolated_b.public_state
        checks.append(CheckResult(f"{task.id}:state_isolation", isolated, "episode mutations remain isolated"))

        replay = AudioEnvironment(package)
        replay.reset(task.id, seed=41, model_id="oracle")
        for action in task.oracle_actions:
            replay.step(action)
        _finish(replay)
        deterministic = (
            oracle.state_snapshot == replay.state_snapshot
            and oracle.trace.final_metrics == replay.trace.final_metrics
            and [s.reward_events for s in oracle.trace.steps] == [s.reward_events for s in replay.trace.steps]
        )
        checks.append(CheckResult(f"{task.id}:deterministic_replay", deterministic, "state and rewards match"))

        if task.required_tools:
            expected = task.required_tools[0]
            mutated_args = dict(expected.arguments)
            first_key = next(iter(mutated_args))
            mutated_args[first_key] = "__mutated__"
            mutant = AudioEnvironment(package)
            mutant.reset(task.id, seed=41, model_id="mutation")
            mutant.step(ToolCallAction(expected.name, mutated_args))
            mutant.step(SpeakAction(b"finish", "finished", True))
            rejected = not bool(mutant.trace.final_metrics.get("terminal_pass"))
            checks.append(CheckResult(f"{task.id}:reward_mutation", rejected, "wrong arguments rejected"))

        if len(task.required_tools) > 1:
            wrong_order = AudioEnvironment(package)
            wrong_order.reset(task.id, seed=41, model_id="wrong-order")
            later = task.required_tools[1]
            wrong_order.step(ToolCallAction(later.name, dict(later.arguments)))
            wrong_order.step(SpeakAction(b"finish", " ".join(task.required_communications), True))
            rejected = not bool(wrong_order.trace.final_metrics.get("terminal_pass"))
            checks.append(CheckResult(f"{task.id}:wrong_order", rejected, "out-of-order milestone rejected"))

    return ValidationReport(tuple(checks))

