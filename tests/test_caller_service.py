from types import SimpleNamespace
from threading import Thread

import pytest

from audioenv.models.caller import CallerTurn
from audioenv.platform.caller_service import (
    FrozenCallerHTTPClient, FrozenCallerSessionManager, create_server,
)


class _FrozenCaller:
    model_id = "fixture/frozen-omni"
    capabilities = SimpleNamespace(trainable=False)

    def __init__(self): self.active_goal = None

    def start_isolated_session(self, private_goal, seed):
        self.active_goal = private_goal
        return CallerTurn(
            f"audio:{private_goal}:{seed}".encode(), "fixture", 0,
            semantic_text=f"opening for {private_goal}",
        ), [{"private_goal": private_goal, "turn": 0}]

    def continue_isolated_session(self, history, agent_audio):
        # The manager must restore the selected session rather than whatever
        # private goal was most recently used by another rollout.
        goal = history[0]["private_goal"]
        turn = history[-1]["turn"] + 1
        return CallerTurn(
            b"reply:" + agent_audio, "fixture", turn,
            semantic_text=f"reply for {goal}",
        ), [*history, {"private_goal": goal, "turn": turn}]


def test_frozen_caller_manager_isolates_private_goals_and_histories():
    manager = FrozenCallerSessionManager(_FrozenCaller())
    first = manager.create("refund order A", 7)
    second = manager.create("exchange order B", 8)
    assert "private_goal" not in first and "history" not in first
    assert first["audio_base64"] != second["audio_base64"]

    reply = manager.respond(first["session_id"], b"agent wav")
    assert reply["semantic_text"] == "reply for refund order A"
    assert manager.sessions[first["session_id"]].history[-1]["turn"] == 1
    assert manager.sessions[second["session_id"]].history[-1]["turn"] == 0
    assert manager.delete(first["session_id"])
    with pytest.raises(KeyError, match="not found"):
        manager.respond(first["session_id"], b"more")


def test_caller_manager_rejects_trainable_model():
    caller = _FrozenCaller()
    caller.capabilities = SimpleNamespace(trainable=True)
    with pytest.raises(ValueError, match="frozen"):
        FrozenCallerSessionManager(caller)


def test_loopback_http_service_round_trip_and_ssrf_guard():
    manager = FrozenCallerSessionManager(_FrozenCaller())
    server = create_server(manager, "127.0.0.1", 0)
    thread = Thread(target=server.serve_forever, daemon=True); thread.start()
    try:
        client = FrozenCallerHTTPClient(f"http://127.0.0.1:{server.server_port}", timeout_seconds=2)
        opened = client.create("refund order A", 7)
        reply = client.respond(opened["session_id"], b"agent wav")
        assert reply["semantic_text"] == "reply for refund order A"
        client.delete(opened["session_id"])
        assert not manager.sessions
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=2)

    with pytest.raises(ValueError, match="loopback"):
        FrozenCallerHTTPClient("http://example.com")
