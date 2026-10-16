from __future__ import annotations

import asyncio
import base64
import io
from pathlib import Path
from types import SimpleNamespace
import wave

import pytest

from audioenv.training.qwen_tau2_runtime import _global_step, run_qwen_tau2_rollout


def _wav(sample: bytes = b"\0\0") -> bytes:
    stream = io.BytesIO()
    with wave.open(stream, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16_000)
        output.writeframes(sample * 160)
    return stream.getvalue()


class _Tau2:
    def __init__(self):
        self.tools = []
        self.spoken = []

    def reset(self, task_id):
        assert task_id == "retail-1"
        return {"session_id": "tau-session", "tool_schemas": [{"name": "lookup_order"}]}

    def caller_context(self, session_id):
        assert session_id == "tau-session"
        return {"private_goal": "refund order A"}

    def tool(self, session_id, name, arguments):
        self.tools.append((session_id, name, arguments))
        return {"result": {"order_id": "A", "eligible": True}, "private_hash": "hidden"}

    def speak(self, session_id, role, semantic_text, *, end_episode=False):
        self.spoken.append((session_id, role, semantic_text, end_episode))
        return {"recorded": True}

    def verify(self, session_id):
        assert session_id == "tau-session"
        return {"upstream_reward": 1.0, "private_details": "not exported"}

    def close(self):
        return None


class _Caller:
    def __init__(self, _url):
        self.deleted = []

    def create(self, private_goal, seed):
        assert "refund order A" in private_goal and seed == 17
        return {
            "session_id": "caller-session",
            "audio_base64": base64.b64encode(_wav()).decode(),
            "semantic_text": "I need help with order A.",
        }

    def respond(self, session_id, agent_audio):
        assert session_id == "caller-session" and agent_audio.startswith(b"RIFF")
        return {
            "audio_base64": base64.b64encode(_wav(b"\1\0")).decode(),
            "semantic_text": "Yes, please continue.",
        }

    def delete(self, session_id):
        self.deleted.append(session_id)


class _Tokenizer:
    outputs = {
        10: '{"tool_call":{"name":"lookup_order","arguments":{"order_id":"A"}}}',
        11: "I found the order. Would you like me to continue?",
        12: '{"speak":"The refund is complete.","terminate":true}',
    }

    def decode(self, token_ids, *, skip_special_tokens):
        assert skip_special_tokens
        return self.outputs[int(token_ids[0])]


class _Server:
    def __init__(self):
        self.index = 0

    async def generate(self, **kwargs):
        assert kwargs["sampling_params"]["logprobs"] == 0
        token = (10, 11, 12)[self.index]
        self.index += 1
        return SimpleNamespace(
            token_ids=[token], log_probs=[-0.1 * self.index], stop_reason="stop",
            extra_fields={
                "audioenv_native_audio_bound": True,
                "audioenv_native_audio": [0.0, 0.1, -0.1, 0.0],
                "audioenv_native_audio_sample_rate": 24_000,
            },
        )


class _AgentLoop:
    def __init__(self):
        self.server_manager = _Server()
        self.tokenizer = _Tokenizer()
        self.index = 0

    async def process_multi_modal_info(self, messages):
        return {"audios": [
            item["audio"]
            for message in messages
            for item in message["content"]
            if item.get("type") == "audio"
        ]}

    def _get_mm_processor_kwargs(self, audios):
        return {"audio_count": len(audios)}

    async def apply_chat_template(self, messages, *, audios, mm_processor_kwargs):
        del messages, audios, mm_processor_kwargs
        prompts = (
            (1, 99, 99, 2),
            (1, 99, 99, 2, 10, 20),
            (1, 99, 99, 2, 10, 20, 11, 99, 99, 30),
        )
        prompt = prompts[self.index]
        self.index += 1
        return prompt


def test_qwen_tau2_runtime_executes_native_multiturn_tool_and_reward_path(
    tmp_path: Path, monkeypatch,
):
    tau2 = _Tau2()
    monkeypatch.setenv("AUDIOENV_TAU2_CHECKOUT", str(tmp_path))
    monkeypatch.setenv("AUDIOENV_CALLER_URL", "http://127.0.0.1:8788")
    monkeypatch.setenv("AUDIOENV_ROLLOUT_AUDIO_DIR", str(tmp_path / "audio"))
    monkeypatch.setattr(
        "audioenv.training.qwen_tau2_runtime.Tau2RetailSourceAdapter.open_worker",
        lambda _self, _checkout, timeout_seconds: tau2,
    )
    caller = _Caller("")
    monkeypatch.setattr(
        "audioenv.training.qwen_tau2_runtime.FrozenCallerHTTPClient",
        lambda _url: caller,
    )
    rollout = asyncio.run(run_qwen_tau2_rollout(
        agent_loop=_AgentLoop(),
        sampling_params={"temperature": 0.8},
        sample={
            "task_id": "retail-1", "experiment_id": "exp-1",
            "environment_hash": "a" * 64, "verifier_hash": "b" * 64,
            "policy_version": "policy-1", "checkpoint_id": "base",
            "model_id": "qwen", "seed": 17, "max_turns": 5,
            "global_steps": 4,
        },
    ))
    assert rollout.reward_score is None
    assert rollout.num_turns == 3
    assert rollout.response_mask == (1, 0, 1, 0, 0, 0, 1)
    assert rollout.response_logprobs == pytest.approx((-0.1, 0.0, -0.2, 0.0, 0.0, 0.0, -0.3))
    assert len(rollout.audio_paths) == 2
    assert rollout.mm_processor_kwargs == {"audio_count": 2}
    assert rollout.extra_fields["audioenv_trusted_reward"]["terminal"] == 1.0
    assert rollout.extra_fields["min_global_steps"] == 4
    assert rollout.extra_fields["max_global_steps"] == 4
    draft = rollout.extra_fields["audioenv_rollout_draft"]
    assert draft["terminal_reward"] == 1.0 and not draft["truncated"]
    assert draft["policy_version"] == "policy-1@verl-step-4"
    assert draft["dataset_policy_version"] == "policy-1"
    assert "private_details" not in repr(draft)
    assert tau2.tools == [("tau-session", "lookup_order", {"order_id": "A"})]
    assert caller.deleted == ["caller-session"]


def test_qwen_tau2_runtime_rejects_missing_or_invalid_actor_step():
    for value in (None, True, -1, "4"):
        with pytest.raises(ValueError, match="global_steps"):
            _global_step({"global_steps": value})
