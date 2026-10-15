import asyncio
from types import SimpleNamespace

import pytest

from audioenv.training.vllm_audio_bridge import install_native_audio_bridge


class _Engine:
    def __init__(self, outputs): self.outputs = outputs
    async def generate(self, **kwargs):
        self.kwargs = kwargs
        for output in self.outputs:
            yield output


class _Server:
    _ar_mode = True

    def __init__(self, outputs): self.engine = _Engine(outputs)

    async def _run_generation(self, prompt, params, request_id, lora_request, priority):
        raise AssertionError("original AR path must be replaced")

    def _process_output(self, final_output, params, sampling_params):
        request = final_output.request_output
        return SimpleNamespace(
            token_ids=request.outputs[0].token_ids,
            log_probs=request.outputs[0].log_probs,
            extra_fields={"original": True},
        )


def _request(*, tokens=(), audio=None):
    multimodal = None if audio is None else {"audio": audio, "sr": 24_000}
    completion = SimpleNamespace(
        token_ids=list(tokens), log_probs=[-0.1] * len(tokens), multimodal_output=multimodal,
    )
    return SimpleNamespace(outputs=[completion])


def test_bridge_binds_same_request_thinker_tokens_and_talker_audio():
    assert install_native_audio_bridge(_Server)
    assert not install_native_audio_bridge(_Server)
    thinker = SimpleNamespace(stage_id=0, request_output=_request(tokens=(10, 11)))
    audio = [0.1, -0.1]
    final = SimpleNamespace(stage_id=2, request_output=_request(audio=audio))
    server = _Server([thinker, final])
    output = asyncio.run(server._run_generation("prompt", ["params"], "request", None, 0))
    tokens = server._process_output(output, None, {})
    assert tokens.token_ids == [10, 11]
    assert tokens.extra_fields["audioenv_native_audio"] is audio
    assert tokens.extra_fields["audioenv_native_audio_sample_rate"] == 24_000
    assert output.request_output is final.request_output


def test_bridge_fails_closed_when_pipeline_drops_either_stage():
    class MissingServer(_Server): pass
    install_native_audio_bridge(MissingServer)
    with pytest.raises(RuntimeError, match="stage-0 Thinker"):
        asyncio.run(MissingServer([
            SimpleNamespace(stage_id=2, request_output=_request(audio=[0.0])),
        ])._run_generation("prompt", [], "request", None, 0))

    thinker = SimpleNamespace(stage_id=0, request_output=_request(tokens=(1,)))
    final = SimpleNamespace(stage_id=2, request_output=_request())
    server = MissingServer([thinker, final])
    output = asyncio.run(server._run_generation("prompt", [], "request", None, 0))
    with pytest.raises(RuntimeError, match="Talker waveform"):
        server._process_output(output, None, {})
