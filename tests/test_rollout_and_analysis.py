import json

from audioenv.analysis import classify_failures
from audioenv.models.base import ModelAction
from audioenv.reference import build_reference_package
from audioenv.rollout import RolloutRequest, RolloutWorker, export_trainer_jsonl
from audioenv.runtime import AudioEnvironment
from audioenv.spec import SpeakAction, ToolCallAction


class QueuedModel:
    model_id = "test/queued-model"

    def __init__(self, actions):
        self.actions = iter(actions)

    def act(self, observation):
        action = next(self.actions)
        return ModelAction(action, (10, 11), (-0.1, -0.2), {"adapter": "test"})


def test_rollout_carries_trainer_metadata_and_exports_jsonl(tmp_path):
    package = build_reference_package()
    task = package.task("retail-refund-001")
    trace = RolloutWorker(package, QueuedModel(task.oracle_actions)).run(
        RolloutRequest(task.id, seed=5)
    )
    assert trace.final_metrics["terminal_pass"] is True
    assert trace.steps[0].training["token_ids"] == [10, 11]
    output = export_trainer_jsonl([trace], tmp_path / "rollouts.jsonl")
    row = json.loads(output.read_text())
    assert row["steps"][0]["training"]["logprobs"] == [-0.1, -0.2]


def test_failure_categories_are_actionable():
    env = AudioEnvironment(build_reference_package())
    env.reset("retail-refund-001")
    bad = ToolCallAction("lookup_order", {"order_id": "misheard"})
    env.step(bad)
    env.step(bad)
    env.step(SpeakAction(b"audio", "done", True))
    assert classify_failures(env.trace) == (
        "dead_loop",
        "incomplete_communication",
        "incorrect_final_state",
        "missing_required_action",
        "wrong_tool_or_arguments",
    )

