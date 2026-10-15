"""Objects loaded by the isolated VeRL worker without importing VeRL at startup."""

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
