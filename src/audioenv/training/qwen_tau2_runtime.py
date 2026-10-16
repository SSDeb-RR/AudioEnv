"""Production tau2/Qwen runtime loaded inside the pinned VeRL worker."""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import io
import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..adapters.tau2 import Tau2RetailSourceAdapter
from ..platform.caller_service import FrozenCallerHTTPClient
from .tau2_episode import PolicyObservation, PolicyTurn, Tau2NativeEpisodeController, parse_episode_action
from .verl_runtime import NativeVerlRollout


async def run_qwen_tau2_rollout(*, agent_loop, sampling_params, sample: dict[str, Any]) -> NativeVerlRollout:
    """Execute one live-policy rollout without exposing tau2 private truth."""

    task_id = _sample_string(sample, "task_id")
    experiment_id = _sample_string(sample, "experiment_id")
    verifier_hash = _hex_hash(sample, "verifier_hash")
    environment_hash = _hex_hash(sample, "environment_hash")
    dataset_policy_version = _sample_string(sample, "policy_version")
    global_step = _global_step(sample)
    # The dataset value identifies the policy lineage; VeRL's global step is
    # the synchronized rollout-weight version for this online update.
    policy_version = f"{dataset_policy_version}@verl-step-{global_step}"
    seed = int(sample.get("seed", 0))
    checkout = Path(_environment("AUDIOENV_TAU2_CHECKOUT")).resolve()
    caller = FrozenCallerHTTPClient(_environment("AUDIOENV_CALLER_URL"))
    tau2 = Tau2RetailSourceAdapter().open_worker(checkout, timeout_seconds=float(os.getenv("AUDIOENV_TAU2_TIMEOUT", "60")))
    audio_dir = Path(os.getenv("AUDIOENV_ROLLOUT_AUDIO_DIR", "/tmp/audioenv-rollout-audio")).resolve()
    audio_dir.mkdir(parents=True, exist_ok=True)

    messages: list[dict[str, Any]] = [{
        "role": "system", "content": [{"type": "text", "text": (
            "You are a voice retail agent. Use exactly one JSON tool_call envelope when a tool is needed. "
            "For speech, respond naturally. To end, emit exactly {\"speak\":\"final words\",\"terminate\":true}."
        )}],
    }]
    prompt_rows: list[tuple[tuple[int, ...], tuple[int, ...], tuple[float, ...]]] = []
    audio_paths: list[str] = []
    audio_artifacts: list[dict[str, str]] = []
    last_audio_hash = None
    last_mm_kwargs: dict[str, Any] = {}
    last_audio_inputs: tuple[Any, ...] = ()

    async def generate(observation: PolicyObservation) -> PolicyTurn:
        nonlocal last_audio_hash, last_mm_kwargs, last_audio_inputs
        digest = sha256(observation.caller_audio).hexdigest()
        if digest != last_audio_hash:
            path = _write_audio(audio_dir, observation.caller_audio, "caller")
            audio_paths.append(str(path))
            audio_artifacts.append({"path": str(path), "producer": "caller"})
            messages.append({"role": "user", "content": [{"type": "audio", "audio": str(path)}]})
            last_audio_hash = digest
        if observation.tool_result is not None:
            messages.append({
                "role": "user", "content": [{"type": "text", "text": "Tool result: " + json.dumps(
                    observation.tool_result, sort_keys=True, separators=(",", ":"),
                )}],
            })
        if observation.turn_index == 1 and observation.tool_schemas:
            messages[0]["content"][0]["text"] += " Available public tools: " + json.dumps(
                observation.tool_schemas, sort_keys=True, separators=(",", ":"),
            )
        multimodal = await agent_loop.process_multi_modal_info(messages)
        # VeRL's generic dataset helper extracts path strings. Qwen's processor
        # requires decoded, 16 kHz waveforms, and trainer-side multimodal
        # reconstruction must receive the exact same values as rollout.
        audios = _load_qwen_audio_inputs(messages)
        multimodal["audios"] = list(audios)
        last_audio_inputs = audios
        mm_kwargs = agent_loop._get_mm_processor_kwargs(audios)
        last_mm_kwargs = dict(mm_kwargs)
        prompt_ids = await agent_loop.apply_chat_template(
            messages, audios=audios, mm_processor_kwargs=mm_kwargs,
        )
        output = await agent_loop.server_manager.generate(
            request_id=uuid4().hex, prompt_ids=prompt_ids,
            sampling_params={**sampling_params, "logprobs": 0},
            audio_data=audios, mm_processor_kwargs=mm_kwargs,
        )
        if not output.token_ids or output.log_probs is None:
            raise RuntimeError("Qwen rollout did not return policy tokens and log-probabilities")
        if len(output.token_ids) != len(output.log_probs):
            raise RuntimeError("Qwen rollout token/log-probability alignment failed")
        semantic = agent_loop.tokenizer.decode(output.token_ids, skip_special_tokens=True).strip()
        native = output.extra_fields.get("audioenv_native_audio")
        if not output.extra_fields.get("audioenv_native_audio_bound") or native is None:
            raise RuntimeError("Qwen rollout did not return same-request Talker audio")
        wav = _waveform_bytes(native, int(output.extra_fields.get("audioenv_native_audio_sample_rate", 24_000)))
        try:
            parsed_action = parse_episode_action(semantic)
        except ValueError:
            parsed_action = None
        if parsed_action is None or parsed_action.kind == "speak":
            agent_path = _write_audio(audio_dir, wav, "agent")
            audio_artifacts.append({"path": str(agent_path), "producer": "agent"})
        prompt_rows.append((tuple(prompt_ids), tuple(output.token_ids), tuple(float(v) for v in output.log_probs)))
        messages.append({"role": "assistant", "content": [{"type": "text", "text": semantic}]})
        return PolicyTurn(semantic_text=semantic, audio=wav, metadata={"stop_reason": output.stop_reason})

    try:
        result = await Tau2NativeEpisodeController(
            tau2, caller, max_turns=int(sample.get("max_turns", 16)),
        ).run(task_id, seed, generate)
    finally:
        tau2.close()
    prompt_ids, response_ids, response_mask, logprobs = _flatten_policy_rows(prompt_rows)
    trusted = {
        "terminal": result.terminal_reward,
        "process": result.process_reward,
        "reward_version": str(sample.get("reward_version", "tau2-official-v1")),
        "verifier_hash": verifier_hash,
    }
    public_trace = {
        "schema_version": "audioenv.native-rollout-draft/v1",
        "experiment_id": experiment_id, "environment_hash": environment_hash,
        "verifier_hash": verifier_hash, "task_id": task_id, "seed": seed,
        "policy_version": policy_version, "truncated": result.truncated,
        "dataset_policy_version": dataset_policy_version,
        "verl_global_step": global_step,
        "truncation_reason": result.truncation_reason,
        "episode_id": uuid4().hex,
        "model_id": str(sample.get("model_id", "Qwen/Qwen3-Omni-30B-A3B-Instruct")),
        "checkpoint_id": str(sample.get("checkpoint_id", "base")),
        "reward_version": trusted["reward_version"],
        "audio_profile_id": str(sample.get("audio_profile_id", "native-clean")),
        "terminal_reward": result.terminal_reward,
        "process_reward": result.process_reward,
        "audio_paths": audio_paths,
        "audio_artifacts": audio_artifacts,
        "actions": [asdict(step.action) for step in result.steps],
    }
    return NativeVerlRollout(
        prompt_ids=prompt_ids, response_ids=response_ids, response_mask=response_mask,
        # Leave reward_score unset so VeRL invokes compute_audioenv_score and
        # validates the trusted payload against the immutable verifier hash.
        response_logprobs=logprobs, reward_score=None,
        num_turns=len(result.steps), audio_inputs=last_audio_inputs,
        audio_paths=tuple(audio_paths),
        mm_processor_kwargs=last_mm_kwargs,
        extra_fields={
            "audioenv_trusted_reward": trusted,
            "audioenv_verifier_hash": verifier_hash,
            "audioenv_rollout_draft": public_trace,
            # Pinned VeRL V1 consumes these fields for trajectory-span and
            # off-policy staleness accounting. This synchronous rollout uses
            # one actor version for the complete episode.
            "min_global_steps": global_step,
            "max_global_steps": global_step,
        },
    )


def _flatten_policy_rows(rows):
    if not rows:
        raise RuntimeError("tau2 episode produced no policy generations")
    initial = rows[0][0]
    response: list[int] = []
    mask: list[int] = []
    logprobs: list[float] = []
    previous = initial
    for index, (prompt, generated, probabilities) in enumerate(rows):
        if index:
            if prompt[: len(previous)] != previous:
                raise RuntimeError("Qwen multi-turn token context is not append-only")
            inserted = prompt[len(previous) :]
            response.extend(inserted); mask.extend([0] * len(inserted)); logprobs.extend([0.0] * len(inserted))
        response.extend(generated); mask.extend([1] * len(generated)); logprobs.extend(probabilities)
        previous = prompt + generated
    return initial, tuple(response), tuple(mask), tuple(logprobs)


def _write_audio(root: Path, payload: bytes, producer: str) -> Path:
    digest = sha256(payload).hexdigest()
    destination = root / producer
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"{digest}.wav"
    if not path.exists():
        path.write_bytes(payload)
    return path


def _load_qwen_audio_inputs(messages: list[dict[str, Any]]) -> tuple[Any, ...]:
    """Decode Qwen conversation audio using the model family's own utility."""

    from qwen_omni_utils import process_audio_info

    decoded = process_audio_info(messages, use_audio_in_video=False)
    if not isinstance(decoded, list) or not decoded:
        raise RuntimeError("Qwen audio preprocessing produced no waveform inputs")
    if any(isinstance(item, (str, bytes, os.PathLike)) for item in decoded):
        raise RuntimeError("Qwen audio preprocessing did not materialize waveform inputs")
    return tuple(decoded)


def _waveform_bytes(waveform: Any, sample_rate: int) -> bytes:
    import soundfile as sf

    if hasattr(waveform, "detach"):
        waveform = waveform.detach().cpu().numpy()
    if hasattr(waveform, "reshape"):
        waveform = waveform.reshape(-1)
    stream = io.BytesIO()
    sf.write(stream, waveform, samplerate=sample_rate, format="WAV")
    return stream.getvalue()


def _sample_string(sample: dict[str, Any], key: str) -> str:
    value = sample.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"rollout sample requires {key}")
    return value


def _hex_hash(sample: dict[str, Any], key: str) -> str:
    value = _sample_string(sample, key)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"rollout sample {key} must be a lowercase sha256")
    return value


def _global_step(sample: dict[str, Any]) -> int:
    value = sample.get("global_steps")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("rollout sample requires a non-negative integer global_steps")
    return value


def _environment(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise RuntimeError(f"{key} is required")
    return value
