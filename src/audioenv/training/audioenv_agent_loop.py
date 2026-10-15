"""AudioEnv rollout bridge for the pinned VeRL-Omni agent-loop boundary.

The bridge is dependency-light so its reward, isolation, and masking invariants
can be tested locally. Registration into VeRL's GPU worker happens only in the
pinned CUDA environment and remains release-gated.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..contracts import TrajectoryV1
from ..rollout_v1 import V1RolloutRequest, V1RolloutWorker


@dataclass(frozen=True, slots=True)
class VerlOmniBatch:
    generated_ids: tuple[tuple[int, ...], ...]
    old_logprobs: tuple[tuple[float, ...], ...]
    response_masks: tuple[tuple[int, ...], ...]
    rewards: tuple[float, ...]
    policy_versions: tuple[str, ...]


class AudioEnvAgentLoop:
    """Run one isolated environment and export only agent Thinker tokens."""

    def __init__(self, worker: V1RolloutWorker):
        self.worker = worker

    def run(self, request: V1RolloutRequest, *, cancel_check=None) -> TrajectoryV1:
        trajectory = self.worker.run(request, cancel_check=cancel_check)
        self.validate_trajectory(trajectory)
        return trajectory

    @staticmethod
    def validate_trajectory(trajectory: TrajectoryV1) -> None:
        for step in trajectory.steps:
            generation = step.generation
            if len(generation.generated_ids) != len(generation.old_logprobs):
                raise ValueError("generated token/log-probability length mismatch")
            if len(generation.generated_ids) != len(generation.action_mask):
                raise ValueError("generated token/action-mask length mismatch")
            if any(mask not in {0, 1} for mask in generation.action_mask):
                raise ValueError("action mask must be binary")
            if generation.policy_version != trajectory.policy_version:
                raise ValueError("policy version drift")

    @staticmethod
    def to_verl_batch(trajectories: tuple[TrajectoryV1, ...]) -> VerlOmniBatch:
        if not trajectories:
            raise ValueError("at least one trajectory is required")
        generated: list[tuple[int, ...]] = []
        logprobs: list[tuple[float, ...]] = []
        masks: list[tuple[int, ...]] = []
        rewards: list[float] = []
        versions: list[str] = []
        for trajectory in trajectories:
            AudioEnvAgentLoop.validate_trajectory(trajectory)
            for step in trajectory.steps:
                generated.append(step.generation.generated_ids)
                logprobs.append(step.generation.old_logprobs)
                masks.append(step.generation.action_mask)
                rewards.append(step.reward)
                versions.append(trajectory.policy_version)
        return VerlOmniBatch(tuple(generated), tuple(logprobs), tuple(masks), tuple(rewards), tuple(versions))
