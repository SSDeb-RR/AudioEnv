"""Dependency-light AudioEnv to VeRL-Omni trajectory bridge."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

from ..contracts import TrajectoryV1, TrajectoryV2
from ..rollout_v1 import V1RolloutRequest, V1RolloutWorker


@dataclass(frozen=True, slots=True)
class VerlOmniBatch:
    input_ids: tuple[tuple[int, ...], ...]
    generated_ids: tuple[tuple[int, ...], ...]
    rollout_logprobs: tuple[tuple[float, ...], ...]
    response_masks: tuple[tuple[int, ...], ...]
    position_ids: tuple[tuple[int, ...], ...]
    rewards: tuple[float, ...]
    group_ids: tuple[str, ...]
    policy_versions: tuple[str, ...]


class AudioEnvAgentLoop:
    def __init__(self, worker: V1RolloutWorker | None = None):
        self.worker = worker

    def run(self, request: V1RolloutRequest, *, cancel_check=None) -> TrajectoryV1:
        if self.worker is None:
            raise RuntimeError("a V1 rollout worker was not configured")
        trajectory = self.worker.run(request, cancel_check=cancel_check)
        self.validate_v1(trajectory)
        return trajectory

    @staticmethod
    def validate_v1(trajectory: TrajectoryV1) -> None:
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
    def to_verl_batch(
        trajectories: tuple[TrajectoryV2, ...], *,
        recompute_logprobs: Callable[[tuple[int, ...], tuple[int, ...]], tuple[float, ...]] | None = None,
        tolerance: float = 5e-3,
    ) -> VerlOmniBatch:
        if not trajectories:
            raise ValueError("at least one trajectory is required")
        rows = []
        versions = {item.policy_version for item in trajectories}
        if len(versions) != 1:
            raise ValueError("batch contains policy-version drift")
        group_sizes: dict[str, int] = {}
        for trajectory in trajectories:
            group_sizes[trajectory.task_id] = group_sizes.get(trajectory.task_id, 0) + 1
            for turn in trajectory.turns:
                if recompute_logprobs:
                    current = recompute_logprobs(turn.input_ids, turn.generated_ids)
                    if len(current) != len(turn.rollout_logprobs):
                        raise ValueError("recomputed log-probability length mismatch")
                    compared = [abs(a - b) for a, b, mask in zip(current, turn.rollout_logprobs, turn.response_mask) if mask]
                    if compared and max(compared) > tolerance:
                        raise ValueError("rollout/training log-probability mismatch")
                rows.append((trajectory, turn, trajectory.terminal_reward + trajectory.process_reward))
        if any(size < 2 for size in group_sizes.values()):
            raise ValueError("GRPO/GSPO requires at least two trajectories per task group")
        if any(not math.isfinite(row[2]) for row in rows):
            raise ValueError("rewards must be finite")
        return VerlOmniBatch(
            input_ids=tuple(turn.input_ids for _, turn, _ in rows),
            generated_ids=tuple(turn.generated_ids for _, turn, _ in rows),
            rollout_logprobs=tuple(turn.rollout_logprobs for _, turn, _ in rows),
            response_masks=tuple(turn.response_mask for _, turn, _ in rows),
            position_ids=tuple(turn.position_ids for _, turn, _ in rows),
            rewards=tuple(reward for _, _, reward in rows),
            group_ids=tuple(trajectory.task_id for trajectory, _, _ in rows),
            policy_versions=tuple(trajectory.policy_version for trajectory, _, _ in rows),
        )

    @staticmethod
    def zero_std_ratio(trajectories: tuple[TrajectoryV2, ...]) -> float:
        groups: dict[str, list[float]] = {}
        for item in trajectories:
            groups.setdefault(item.task_id, []).append(item.terminal_reward + item.process_reward)
        if not groups:
            raise ValueError("at least one reward group is required")
        zero = sum(1 for rewards in groups.values() if len(rewards) < 2 or max(rewards) == min(rewards))
        return zero / len(groups)
