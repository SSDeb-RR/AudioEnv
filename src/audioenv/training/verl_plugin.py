"""Dependency-light functions loaded inside the isolated VeRL worker.

The control plane never imports VeRL.  VeRL loads this module through
``VERL_USE_EXTERNAL_MODULES`` and resolves ``compute_audioenv_score`` using its
documented custom reward-function boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


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


class AudioEnvVerlAgentLoop:
    def __init__(self, rollout_factory: Callable[..., Any]):
        self.rollout_factory = rollout_factory

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> dict[str, Any]:
        trajectory = await self.rollout_factory(sampling_params=sampling_params, **kwargs)
        return {
            "trajectory": trajectory.model_dump(mode="json"),
            "audioenv_trusted_reward": {
                "terminal": trajectory.terminal_reward, "process": trajectory.process_reward,
                "reward_version": trajectory.reward_version, "verifier_hash": trajectory.verifier_hash,
            },
        }
