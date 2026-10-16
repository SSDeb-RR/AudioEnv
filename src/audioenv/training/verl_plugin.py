"""Dependency-light functions loaded inside the isolated VeRL worker.

The control plane never imports VeRL.  VeRL loads this module through
``VERL_USE_EXTERNAL_MODULES`` and resolves ``compute_audioenv_score`` using its
documented custom reward-function boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import os
from typing import Any, Callable

from ..contracts import TrajectoryV2
from .verl_runtime import NativeVerlRollout

if os.environ.get("AUDIOENV_ENABLE_NATIVE_AUDIO_BRIDGE") == "1":
    from .vllm_audio_bridge import install_native_audio_bridge
    from .verl_trajectory_bridge import install_trajectory_finalize_bridge

    install_native_audio_bridge()
    install_trajectory_finalize_bridge()

try:  # Imported only inside the pinned CUDA worker environment.
    from verl.experimental.agent_loop.agent_loop import (
        AgentLoopBase as _VerlAgentLoopBase,
        AgentLoopMetrics as _VerlAgentLoopMetrics,
        AgentLoopOutput as _VerlAgentLoopOutput,
        register as _verl_register,
    )
except ImportError:  # Local control-plane installs intentionally omit VeRL.
    _VerlAgentLoopBase = object
    _VerlAgentLoopMetrics = None
    _VerlAgentLoopOutput = None

    def _verl_register(_: str):
        return lambda cls: cls


@dataclass(slots=True)
class AudioEnvRewardManager:
    terminal_weight: float = 1.0
    process_weight: float = 1.0

    def __call__(self, sample: dict[str, Any]) -> float:
        trusted = sample.get("audioenv_trusted_reward")
        if not isinstance(trusted, dict):
            raise ValueError("missing protected AudioEnv reward payload")
        return self.terminal_weight * float(trusted.get("terminal", 0.0)) + self.process_weight * float(trusted.get("process", 0.0))


def compute_audioenv_score(
    data_source: str,
    solution_str: str,
    ground_truth: object,
    extra_info: dict[str, Any],
    **_: Any,
) -> dict[str, float | str]:
    """Return only the reward produced by AudioEnv's protected verifier.

    ``solution_str`` and ``ground_truth`` are deliberately ignored: a model may
    not earn reward by printing a claimed answer, and trusted verifier material
    must not be embedded in the public prompt.  The AudioEnv agent loop places
    the verifier result in ``extra_info`` after the environment terminates.
    """

    del solution_str, ground_truth
    trusted = extra_info.get("audioenv_trusted_reward")
    if not isinstance(trusted, dict):
        raise ValueError("missing protected AudioEnv reward payload")
    terminal = float(trusted.get("terminal", 0.0))
    process = float(trusted.get("process", 0.0))
    terminal_weight = float(extra_info.get("audioenv_terminal_weight", 1.0))
    process_weight = float(extra_info.get("audioenv_process_weight", 1.0))
    verifier_hash = trusted.get("verifier_hash")
    expected_hash = extra_info.get("audioenv_verifier_hash")
    if not verifier_hash or verifier_hash != expected_hash:
        raise ValueError("protected verifier hash mismatch")
    return {
        "score": terminal_weight * terminal + process_weight * process,
        "terminal_reward": terminal,
        "process_reward": process,
        "data_source": str(data_source),
    }


@dataclass(frozen=True, slots=True)
class VerlAgentLoopPayload:
    prompt_ids: tuple[int, ...]
    response_ids: tuple[int, ...]
    response_mask: tuple[int, ...]
    response_logprobs: tuple[float, ...]
    reward_score: float | None
    num_turns: int
    extra_fields: dict[str, Any]


def trajectory_to_verl_payload(trajectory: TrajectoryV2) -> VerlAgentLoopPayload:
    """Convert a V2 trace into VeRL's single sequence without hiding drift.

    Every later turn must extend the prior prompt plus generated action. Tokens
    introduced by caller audio or tool results are retained with mask zero;
    only policy-generated Thinker tokens receive mask one.
    """

    first = trajectory.turns[0]
    prompt = tuple(first.input_ids)
    response_ids: list[int] = []
    response_mask: list[int] = []
    response_logprobs: list[float] = []
    previous_context = prompt
    for index, turn in enumerate(trajectory.turns):
        current = tuple(turn.input_ids)
        if index:
            if current[: len(previous_context)] != previous_context:
                raise ValueError("multi-turn context is not an append-only policy lineage")
            observation_tokens = current[len(previous_context) :]
            response_ids.extend(observation_tokens)
            response_mask.extend([0] * len(observation_tokens))
            response_logprobs.extend([0.0] * len(observation_tokens))
        response_ids.extend(turn.generated_ids)
        response_mask.extend(turn.response_mask)
        response_logprobs.extend(turn.rollout_logprobs)
        previous_context = current + tuple(turn.generated_ids)
    return VerlAgentLoopPayload(
        prompt_ids=prompt,
        response_ids=tuple(response_ids),
        response_mask=tuple(response_mask),
        response_logprobs=tuple(response_logprobs),
        # VeRL must recompute this through compute_audioenv_score. If a score
        # is returned here, its agent-loop batching creates rm_scores and the
        # configured reward manager deliberately skips the custom scorer.
        reward_score=None,
        num_turns=len(trajectory.turns),
        extra_fields={
            "audioenv_trusted_reward": {
                "terminal": trajectory.terminal_reward,
                "process": trajectory.process_reward,
                "reward_version": trajectory.reward_version,
                "verifier_hash": trajectory.verifier_hash,
            },
            "audioenv_verifier_hash": trajectory.verifier_hash,
            "audioenv_trajectory": trajectory.model_dump(mode="json"),
        },
    )


def _load_factory(reference: str) -> Callable[..., Any]:
    module_name, separator, attribute = reference.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("rollout factory must use module:attribute syntax")
    factory = getattr(importlib.import_module(module_name), attribute)
    if not callable(factory):
        raise TypeError("configured rollout factory is not callable")
    return factory


@_verl_register("audioenv_agent")
class AudioEnvVerlAgentLoop(_VerlAgentLoopBase):
    """Pinned VeRL AgentLoopBase integration loaded by external modules.

    The factory executes AudioEnv's protected environment/caller loop and must
    return ``TrajectoryV2``. It receives this loop instance so it can use the
    pinned server manager, processor, and tokenizer rather than loading a
    second policy. Local tests may inject a factory directly.
    """

    def __init__(self, *args, rollout_factory: Callable[..., Any] | None = None, **kwargs):
        if _VerlAgentLoopOutput is not None:
            super().__init__(*args, **kwargs)
        self.rollout_factory = rollout_factory
        if self.rollout_factory is None:
            reference = os.environ.get("AUDIOENV_VERL_ROLLOUT_FACTORY", "")
            if not reference:
                raise RuntimeError("AUDIOENV_VERL_ROLLOUT_FACTORY is required")
            self.rollout_factory = _load_factory(reference)

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> dict[str, Any]:
        trajectory = await self.rollout_factory(
            agent_loop=self, sampling_params=sampling_params, **kwargs,
        )
        if isinstance(trajectory, TrajectoryV2):
            payload = trajectory_to_verl_payload(trajectory)
            multi_modal_data = None
            mm_processor_kwargs = None
        elif isinstance(trajectory, NativeVerlRollout):
            payload = VerlAgentLoopPayload(
                prompt_ids=trajectory.prompt_ids, response_ids=trajectory.response_ids,
                response_mask=trajectory.response_mask,
                response_logprobs=trajectory.response_logprobs,
                reward_score=trajectory.reward_score, num_turns=trajectory.num_turns,
                extra_fields=trajectory.extra_fields,
            )
            multi_modal_data = {"audios": list(trajectory.audio_paths)}
            mm_processor_kwargs = trajectory.mm_processor_kwargs
        else:
            raise TypeError("AudioEnv VeRL rollout factory must return TrajectoryV2 or NativeVerlRollout")
        if _VerlAgentLoopOutput is None:
            return {
                "prompt_ids": list(payload.prompt_ids), "response_ids": list(payload.response_ids),
                "response_mask": list(payload.response_mask),
                "response_logprobs": list(payload.response_logprobs),
                "reward_score": payload.reward_score, "num_turns": payload.num_turns,
                "extra_fields": payload.extra_fields, "multi_modal_data": multi_modal_data,
                "mm_processor_kwargs": mm_processor_kwargs,
            }
        return _VerlAgentLoopOutput(
            prompt_ids=list(payload.prompt_ids), response_ids=list(payload.response_ids),
            response_mask=list(payload.response_mask),
            response_logprobs=list(payload.response_logprobs),
            reward_score=payload.reward_score, num_turns=payload.num_turns,
            metrics=_VerlAgentLoopMetrics(), extra_fields=payload.extra_fields,
            multi_modal_data=multi_modal_data, mm_processor_kwargs=mm_processor_kwargs,
        )
