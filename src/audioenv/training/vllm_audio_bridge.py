"""Pinned vLLM-Omni bridge retaining Thinker tokens and Talker waveform.

VeRL-Omni's AR server currently keeps only the last output yielded by a
multi-stage pipeline. In Qwen3-Omni full mode that is the Code2Wav output, so
the stage-0 policy tokens/log-probabilities are lost. This narrow patch keeps
the stage-0 request output on the final object and lets VeRL build its normal
``TokenOutput`` from it while attaching the waveform from the same request.

The bridge is intentionally tied to AudioEnv's pinned VeRL-Omni/vLLM-Omni
revisions and fails closed if either side of the native output is absent.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


PINNED_VLLM_OMNI_COMMIT = "fe478a95ab39e80085da06e3401c3ed471898a57"
_THINKER_OUTPUT = "_audioenv_thinker_request_output"


def install_native_audio_bridge(server_class=None) -> bool:
    """Install the bridge once and return whether this call changed the class."""

    if server_class is None:
        from verl_omni.workers.rollout.vllm_rollout.vllm_omni_async_server import (
            vLLMOmniHttpServer,
        )

        server_class = vLLMOmniHttpServer
    if getattr(server_class, "_audioenv_native_bridge", False):
        return False

    original_run = server_class._run_generation
    original_process = server_class._process_output

    async def run_generation(self, prompt, params, request_id, lora_request, priority):
        if not getattr(self, "_ar_mode", False):
            return await original_run(self, prompt, params, request_id, lora_request, priority)
        generator = self.engine.generate(
            prompt=prompt,
            sampling_params_list=params,
            request_id=request_id,
            lora_request=lora_request,
            priority=priority,
        )
        thinker_output = None
        final_output = None
        async for output in generator:
            final_output = output
            if getattr(output, "stage_id", None) == 0 and getattr(output, "request_output", None) is not None:
                thinker_output = output.request_output
        if final_output is None:
            raise RuntimeError("native Qwen pipeline yielded no output")
        if thinker_output is None:
            raise RuntimeError("native Qwen pipeline did not expose stage-0 Thinker output")
        setattr(final_output, _THINKER_OUTPUT, thinker_output)
        return final_output

    def process_output(self, final_output, params, sampling_params):
        if not getattr(self, "_ar_mode", False):
            return original_process(self, final_output, params, sampling_params)
        thinker_output = getattr(final_output, _THINKER_OUTPUT, None)
        if thinker_output is None:
            raise RuntimeError("AudioEnv native rollout is missing bound Thinker output")
        audio_request = getattr(final_output, "request_output", None)
        waveform, sample_rate = _extract_audio(final_output)
        if waveform is None:
            raise RuntimeError("AudioEnv native rollout is missing bound Talker waveform")
        final_output.request_output = thinker_output
        try:
            token_output = original_process(self, final_output, params, sampling_params)
        finally:
            final_output.request_output = audio_request
        token_output.extra_fields.update({
            "audioenv_native_audio": waveform,
            "audioenv_native_audio_sample_rate": sample_rate,
            "audioenv_native_audio_bound": True,
        })
        return token_output

    server_class._run_generation = run_generation
    server_class._process_output = process_output
    server_class._audioenv_native_bridge = True
    return True


def _extract_audio(final_output: Any) -> tuple[Any | None, int]:
    """Extract one request's waveform from the pinned Omni output contract.

    ``OmniRequestOutput.multimodal_output`` is the public bridge across both
    pipeline and diffusion modes. At the pinned vLLM-Omni revision it is a
    ``MultimodalPayload`` (a ``Mapping``, not a ``dict``), and Qwen Code2Wav
    names its waveform field ``model_outputs``. Values may still be wrapped in
    a single-item sequence after per-request output processing.
    """

    payload = getattr(final_output, "multimodal_output", None)
    if not isinstance(payload, Mapping):
        return None, 0
    waveform = payload.get("audio", payload.get("model_outputs"))
    sample_rate = payload.get("sr", payload.get("sample_rate", 24_000))
    waveform = _unwrap_single_request_value(waveform)
    sample_rate = _unwrap_single_request_value(sample_rate)
    if hasattr(sample_rate, "item"):
        sample_rate = sample_rate.item()
    try:
        sample_rate_value = int(sample_rate)
    except (TypeError, ValueError):
        return None, 0
    if waveform is None or sample_rate_value <= 0:
        return None, 0
    return waveform, sample_rate_value


def _unwrap_single_request_value(value: Any) -> Any:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) != 1:
            return None
        return value[0]
    return value
