from audioenv.models import Qwen3OmniAgentAdapter, Qwen3OmniCallerAdapter, ScriptedAudioCallerAdapter


def test_qwen_probe_does_not_imply_identifier_is_available(tmp_path):
    probe = Qwen3OmniAgentAdapter(tmp_path / "missing-checkpoint").probe()
    assert not probe.available
    assert any("checkpoint" in reason for reason in probe.reasons)


def test_qwen_caller_is_always_frozen(tmp_path):
    caller = Qwen3OmniCallerAdapter(tmp_path / "missing", private_goal="obtain a refund")
    assert caller.capabilities.audio_input
    assert caller.capabilities.audio_output
    assert not caller.capabilities.trainable
    assert caller.capabilities.trainable_modules == ()


def test_scripted_caller_is_deterministic_and_truthfully_labeled():
    first = ScriptedAudioCallerAdapter(("hello", "next"))
    second = ScriptedAudioCallerAdapter(("hello", "next"))
    assert first.reset(3).audio == second.reset(3).audio
    assert "not realistic" in first.probe().reasons[0]
    assert first.respond(b"agent").turn_index == 1
    assert first.respond(b"agent") is None
