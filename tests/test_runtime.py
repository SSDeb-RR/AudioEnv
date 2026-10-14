from concurrent.futures import ThreadPoolExecutor

import pytest

from audioenv.reference import build_reference_package
from audioenv.runtime import AudioEnvironment
from audioenv.spec import SpeakAction, ToolCallAction


def run_oracle(task_id: str):
    package = build_reference_package()
    task = package.task(task_id)
    env = AudioEnvironment(package)
    initial = env.reset(task.id, seed=9, model_id="oracle")
    for action in task.oracle_actions:
        final = env.step(action)
    return initial, final, env


@pytest.mark.parametrize("task_id", ["retail-refund-001", "telecom-roaming-001"])
def test_oracle_passes(task_id):
    initial, final, env = run_oracle(task_id)
    assert initial.audio.startswith(b"RIFF")
    assert final.done
    assert final.metrics["terminal_pass"] is True
    assert env.trace.final_metrics["terminal_pass"] is True


@pytest.mark.parametrize("task_id", ["retail-refund-001", "telecom-roaming-001"])
def test_noop_fails(task_id):
    env = AudioEnvironment(build_reference_package())
    env.reset(task_id)
    final = env.step(SpeakAction(b"audio", "Cannot help", True))
    assert final.metrics["terminal_pass"] is False


def test_wrong_exact_slot_is_not_rewarded():
    env = AudioEnvironment(build_reference_package())
    env.reset("retail-refund-001")
    result = env.step(ToolCallAction("lookup_order", {"order_id": "A-104Z"}))
    assert result.reward == -0.1
    assert result.tool_result == {"error": "not_found"}


def test_successful_but_unrequired_call_gets_no_process_reward():
    env = AudioEnvironment(build_reference_package())
    env.reset("retail-refund-001")
    # The call is valid in the package but irrelevant to the retail task.
    result = env.step(ToolCallAction("lookup_line", {"phone_number": "+1-202-555-0198"}))
    assert result.reward == -0.1  # handler rejects because this episode has no telecom state


def test_repeated_failures_receive_extra_penalty():
    env = AudioEnvironment(build_reference_package())
    env.reset("retail-refund-001")
    action = ToolCallAction("lookup_order", {"order_id": "wrong"})
    assert env.step(action).reward == -0.1
    assert env.step(action).reward == pytest.approx(-0.2)


def test_forbidden_mutation_is_blocked():
    env = AudioEnvironment(build_reference_package())
    env.reset("telecom-roaming-001")
    before = env.state_snapshot
    result = env.step(
        ToolCallAction("change_plan", {"phone_number": "+1-202-555-0198", "plan": "premium"})
    )
    assert result.reward == -0.5
    assert env.state_snapshot == before


def test_agent_view_has_no_transcript_or_hidden_state():
    package = build_reference_package()
    task = package.task("retail-refund-001")
    observation = AudioEnvironment(package).reset(task.id)
    view = observation.agent_view()
    serialized = str(view)
    assert task.initial_utterance not in serialized
    assert "hidden_facts" not in serialized
    assert "expected_state" not in serialized


def test_parallel_episodes_are_isolated():
    package = build_reference_package()

    def mutate_or_not(run_oracle_actions: bool):
        task = package.task("retail-refund-001")
        env = AudioEnvironment(package)
        env.reset(task.id)
        if run_oracle_actions:
            for action in task.oracle_actions:
                env.step(action)
        return env.state_snapshot

    with ThreadPoolExecutor(max_workers=2) as pool:
        mutated, untouched = pool.map(mutate_or_not, (True, False))
    assert mutated["orders"]["A-1042"]["status"] == "refunded"
    assert untouched["orders"]["A-1042"]["status"] == "delivered"

