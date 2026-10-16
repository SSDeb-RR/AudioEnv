"""Finalize native rollout drafts after VeRL computes Qwen positions."""

from __future__ import annotations

from hashlib import sha256
import io
import json
import os
from pathlib import Path
import wave
from typing import Any

from ..contracts import AudioArtifactV1, MultimodalFeatureRefV1, TrainingTurnV2, TrajectoryV2


def install_trajectory_finalize_bridge(worker_class=None, *, audio_token_id_resolver=None) -> bool:
    """Patch VeRL V1's queue worker so traces use processor-derived positions."""

    if worker_class is None:
        from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopWorkerTQ

        worker_class = AgentLoopWorkerTQ
    if getattr(worker_class, "_audioenv_trajectory_bridge", False):
        return False
    original = worker_class._agent_loop_postprocess
    if audio_token_id_resolver is None:
        from verl.experimental.agent_loop.agent_loop import get_processor_token_id

        audio_token_id_resolver = get_processor_token_id

    async def postprocess(self, output, validate, **kwargs):
        outputs = output if isinstance(output, list) else [output]
        for item in outputs:
            draft = item.extra_fields.get("audioenv_rollout_draft")
            if draft is None:
                continue
            import torch

            prompt_ids = tuple(item.prompt_ids)
            response_ids = tuple(item.response_ids)
            input_ids = torch.tensor(prompt_ids + response_ids, dtype=torch.int64)
            attention_mask = torch.ones_like(input_ids, dtype=torch.int64)
            multi_modal_inputs = self._compute_multi_modal_inputs(item, input_ids)
            positions = self._compute_position_ids(
                input_ids.unsqueeze(0), attention_mask.unsqueeze(0), multi_modal_inputs,
            )
            trajectory = finalize_native_trajectory(
                draft=draft,
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                response_mask=tuple(item.response_mask),
                response_logprobs=tuple(item.response_logprobs or ()),
                padded_position_ids=positions,
                prompt_width=len(prompt_ids),
                audio_token_id=audio_token_id_resolver(self.processor, "audio"),
            )
            item.extra_fields["audioenv_trajectory"] = trajectory.model_dump(mode="json")
            _persist_trajectory(trajectory)
        # Finalize before queue insertion: invalid lineage must never enter a
        # training batch. The pinned TQ worker returns None after writing.
        return await original(self, output, validate, **kwargs)

    worker_class._agent_loop_postprocess = postprocess
    worker_class._audioenv_trajectory_bridge = True
    return True


def _persist_trajectory(trajectory: TrajectoryV2) -> Path | None:
    root = os.environ.get("AUDIOENV_TRAJECTORY_DIR")
    if not root:
        return None
    destination = Path(root).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"trajectory-{os.getpid()}.jsonl"
    line = json.dumps(trajectory.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    with path.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
    return path


def finalize_native_trajectory(
    *, draft: dict[str, Any], prompt_ids: tuple[int, ...], response_ids: tuple[int, ...],
    response_mask: tuple[int, ...], response_logprobs: tuple[float, ...],
    padded_position_ids: Any, prompt_width: int, audio_token_id: int | None,
) -> TrajectoryV2:
    """Bind VeRL processor-derived positions and audio lineage to TrajectoryV2."""

    if draft.get("schema_version") != "audioenv.native-rollout-draft/v1":
        raise ValueError("unsupported native rollout draft")
    if len(response_ids) != len(response_logprobs):
        raise ValueError("finalized rollout is missing log-probabilities")
    selected = _select_unpadded_positions(
        padded_position_ids, prompt_width, len(prompt_ids), len(response_ids),
    )
    descriptions = draft.get("audio_artifacts") or [
        {"path": item, "producer": "caller"} for item in draft.get("audio_paths", ())
    ]
    audio = tuple(_audio_artifact(Path(item["path"]), str(item["producer"])) for item in descriptions)
    if not audio:
        raise ValueError("native rollout draft has no audio lineage")
    caller_audio = tuple(item for item in audio if item.producer == "caller")
    # Later caller observations are flattened into response_ids with mask 0.
    # Multimodal lineage therefore spans the complete unpadded policy sequence,
    # not only the first-turn prompt.
    spans = _audio_spans(prompt_ids + response_ids, audio_token_id, len(caller_audio))
    features = tuple(MultimodalFeatureRefV1(
        kind="audio", artifact_id=item.artifact_id, start_token=start, end_token=end,
    ) for item, (start, end) in zip(caller_audio, spans))
    truncated = bool(draft.get("truncated"))
    turn = TrainingTurnV2(
        index=1, input_ids=prompt_ids, generated_ids=response_ids,
        attention_mask=(1,) * (len(prompt_ids) + len(response_ids)),
        response_mask=response_mask, position_ids=selected,
        rollout_logprobs=response_logprobs, multimodal_features=features,
        action={"kind": "episode", "actions": list(draft.get("actions") or ())},
        reward_events=(), done=not truncated,
    )
    return TrajectoryV2(
        experiment_id=_required(draft, "experiment_id"),
        environment_hash=_required(draft, "environment_hash"),
        verifier_hash=_required(draft, "verifier_hash"),
        reward_version=_required(draft, "reward_version"),
        task_id=_required(draft, "task_id"), episode_id=_required(draft, "episode_id"),
        seed=int(draft.get("seed", 0)), model_id=_required(draft, "model_id"),
        checkpoint_id=_required(draft, "checkpoint_id"),
        policy_version=_required(draft, "policy_version"),
        processor_config={"positions": "verl-qwen-processor-derived"},
        audio_artifacts=audio, turns=(turn,),
        terminal_reward=float(draft.get("terminal_reward", draft.get("reward", 0.0))),
        process_reward=float(draft.get("process_reward", 0.0)),
        done=not truncated,
        truncation_reason=str(draft.get("truncation_reason")) if truncated else None,
    )


def _select_unpadded_positions(values: Any, prompt_width: int, prompt_length: int, response_length: int):
    if prompt_width < prompt_length:
        raise ValueError("padded prompt width is smaller than the prompt")
    indices = list(range(prompt_width - prompt_length, prompt_width)) + list(
        range(prompt_width, prompt_width + response_length)
    )
    tensor = values
    if hasattr(tensor, "ndim") and tensor.ndim == 3:
        if int(tensor.shape[0]) != 1:
            raise ValueError("position data must contain exactly one rollout row")
        tensor = tensor[0]
    if hasattr(tensor, "ndim") and tensor.ndim == 2:
        return tuple(tuple(int(channel[index]) for index in indices) for channel in tensor)
    return tuple(int(tensor[index]) for index in indices)


def _audio_spans(prompt_ids: tuple[int, ...], token_id: int | None, expected: int):
    if token_id is None:
        raise ValueError("Qwen processor does not declare an audio token id")
    spans: list[tuple[int, int]] = []
    start = None
    for index, value in enumerate((*prompt_ids, None)):
        if value == token_id and start is None:
            start = index
        elif value != token_id and start is not None:
            spans.append((start, index)); start = None
    if len(spans) != expected:
        raise ValueError(f"expected {expected} audio token spans, found {len(spans)}")
    return tuple(spans)


def _audio_artifact(path: Path, producer: str) -> AudioArtifactV1:
    payload = path.read_bytes()
    digest = sha256(payload).hexdigest()
    with wave.open(io.BytesIO(payload), "rb") as source:
        sample_rate, frame_count = source.getframerate(), source.getnframes()
    return AudioArtifactV1(
        artifact_id=digest, sha256=digest, media_type="audio/wav",
        sample_rate_hz=sample_rate, duration_ms=frame_count * 1000.0 / sample_rate,
        producer=producer,
    )


def _required(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"native rollout draft requires {key}")
    return value
