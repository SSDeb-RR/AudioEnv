"""Local rollout worker and trainer-neutral JSONL export."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
from pathlib import Path

from .adapter import EnvironmentPackage
from .models.base import NativeAudioModelAdapter
from .runtime import AudioEnvironment
from .trace import EpisodeTrace


@dataclass(frozen=True, slots=True)
class RolloutRequest:
    task_id: str
    seed: int
    audio_profile_id: str | None = None


class RolloutWorker:
    def __init__(self, package: EnvironmentPackage, model: NativeAudioModelAdapter):
        self.package = package
        self.model = model

    def run(self, request: RolloutRequest) -> EpisodeTrace:
        env = AudioEnvironment(self.package)
        observation = env.reset(
            request.task_id,
            request.seed,
            audio_profile_id=request.audio_profile_id,
            model_id=self.model.model_id,
        )
        while not observation.done:
            model_action = self.model.act(observation)
            observation = env.step(
                model_action.action,
                training={
                    "token_ids": list(model_action.token_ids),
                    "logprobs": list(model_action.logprobs),
                    **model_action.metadata,
                },
            )
        return env.trace


def run_parallel(
    package: EnvironmentPackage,
    model_factory,
    requests: list[RolloutRequest],
    *,
    workers: int = 4,
) -> list[EpisodeTrace]:
    def execute(request: RolloutRequest) -> EpisodeTrace:
        return RolloutWorker(package, model_factory()).run(request)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(execute, requests))


def export_trainer_jsonl(traces: list[EpisodeTrace], destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        for trace in traces:
            stream.write(json.dumps(trace.to_dict(include_observability=False), sort_keys=True) + "\n")
    return destination

