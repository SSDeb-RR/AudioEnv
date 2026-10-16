import base64
import asyncio
import json

import pytest

from audioenv.training.tau2_episode import (
    PolicyTurn, Tau2NativeEpisodeController, parse_episode_action,
)


class _Tau2:
    def __init__(self): self.events = []
    def reset(self, task_id):
        self.events.append(("reset", task_id))
        return {"session_id": "tau", "tool_schemas": [{"name": "lookup"}]}
    def caller_context(self, session_id):
        self.events.append(("private", session_id)); return {"user_scenario": {"hidden": "refund A"}}
    def speak(self, session_id, role, semantic_text, *, end_episode=False):
        self.events.append(("speak", role, semantic_text, end_episode)); return {"recorded": True}
    def tool(self, session_id, name, arguments):
        self.events.append(("tool", name, arguments)); return {"result": {"order": "A"}, "db_hash": "private"}
    def verify(self, session_id):
        self.events.append(("verify", session_id)); return {"upstream_reward": 1.0, "reward_info": {}}


class _Caller:
    def __init__(self): self.private_goal = None; self.deleted = []
    def create(self, private_goal, seed):
        self.private_goal = private_goal
        return {"session_id": "caller", "audio_base64": base64.b64encode(b"opening wav").decode(), "semantic_text": "I need help"}
    def respond(self, session_id, agent_audio):
        assert agent_audio == b"agent wav"
        return {"audio_base64": base64.b64encode(b"reply wav").decode(), "semantic_text": "Yes"}
    def delete(self, session_id): self.deleted.append(session_id)


def test_tau2_episode_keeps_private_goal_out_of_policy_and_uses_official_reward():
    tau2, caller = _Tau2(), _Caller()
    observations = []
    outputs = iter((
        PolicyTurn('{"tool_call":{"name":"lookup","arguments":{"id":"A"}}}', b"ignored", {"tokens": [1]}),
        PolicyTurn("I can refund that.", b"agent wav", {"tokens": [2]}),
        PolicyTurn('{"speak":"Your refund is complete.","terminate":true}', b"final wav", {"tokens": [3]}),
    ))

    def generate(observation):
        observations.append(observation)
        return next(outputs)

    result = asyncio.run(Tau2NativeEpisodeController(tau2, caller).run("0", 7, generate))
    assert result.terminal_reward == 1 and not result.truncated
    assert result.process_reward == 0
    assert [step.action.kind for step in result.steps] == ["tool", "speak", "speak"]
    assert observations[0].caller_audio == b"opening wav"
    assert observations[0].tool_result is None
    assert observations[0].tool_schemas == ({"name": "lookup"},)
    assert observations[1].tool_result == {"result": {"order": "A"}}
    assert "private" not in repr(observations)
    assert observations[2].caller_audio == b"reply wav"
    assert all("refund A" not in repr(item) for item in observations)
    assert json.loads(caller.private_goal)["user_scenario"]["hidden"] == "refund A"
    assert caller.deleted == ["caller"]
    assert tau2.events[-1] == ("verify", "tau")


def test_tau2_episode_verifies_truncated_run_and_cleans_up():
    tau2, caller = _Tau2(), _Caller()
    result = asyncio.run(Tau2NativeEpisodeController(tau2, caller, max_turns=1).run(
        "0", 1, lambda _: PolicyTurn('{"tool_call":{"name":"lookup","arguments":{}}}', b"", None),
    ))
    assert result.truncated and result.truncation_reason == "max_turns"
    assert caller.deleted == ["caller"]


def test_action_parser_rejects_ambiguous_or_malformed_envelopes():
    assert parse_episode_action("hello").kind == "speak"
    with pytest.raises(ValueError, match="malformed"):
        parse_episode_action("{broken")
    with pytest.raises(ValueError, match="unsupported"):
        parse_episode_action('{"tool_call":{"name":"x","arguments":{}},"speak":"also"}')


def test_tau2_episode_recovers_from_malformed_action_with_repeat_penalty():
    tau2, caller = _Tau2(), _Caller()
    outputs = iter((
        PolicyTurn("{broken", b"bad wav", None),
        PolicyTurn("{broken", b"bad wav", None),
        PolicyTurn('{"speak":"Recovered.","terminate":true}', b"final wav", None),
    ))
    observations = []

    def generate(observation):
        observations.append(observation)
        return next(outputs)

    result = asyncio.run(Tau2NativeEpisodeController(tau2, caller).run("0", 7, generate))
    assert [step.action.kind for step in result.steps] == ["invalid", "invalid", "speak"]
    assert observations[1].tool_result == {"result": {"error": "invalid_action"}}
    assert observations[2].tool_result == {"result": {"error": "invalid_action"}}
    assert result.process_reward == pytest.approx(-0.3)
    assert result.terminal_reward == 1


def test_tau2_episode_turns_tool_worker_failure_into_recovery_observation():
    class RejectingTau2(_Tau2):
        def tool(self, session_id, name, arguments):
            raise RuntimeError("private worker detail")

    tau2, caller = RejectingTau2(), _Caller()
    outputs = iter((
        PolicyTurn('{"tool_call":{"name":"unknown","arguments":{}}}', b"ignored", None),
        PolicyTurn('{"speak":"I could not complete that.","terminate":true}', b"final wav", None),
    ))
    observations = []

    def generate(observation):
        observations.append(observation)
        return next(outputs)

    result = asyncio.run(Tau2NativeEpisodeController(tau2, caller).run("0", 7, generate))
    assert observations[1].tool_result == {"result": {"error": "tool_execution_failed"}}
    assert "private worker detail" not in repr(observations)
    assert result.process_reward == pytest.approx(-0.1)
