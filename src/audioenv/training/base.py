"""Optimizer-neutral trainer contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Protocol

from ..contracts import ResourceEstimate, TrainerCapabilities


MetricSink = Callable[[int, str, float, str], None]
CancelCheck = Callable[[], bool]
EpisodeSink = Callable[[int, dict[str, Any]], None]


class TrainerPlugin(Protocol):
    trainer_id: str

    @property
    def capabilities(self) -> TrainerCapabilities: ...

    def validate_config(self, config: Any) -> tuple[str, ...]: ...

    def preflight(self, config: Any) -> ResourceEstimate: ...

    def train(
        self,
        config: Any,
        destination: Path,
        *,
        metric_sink: MetricSink | None = None,
        cancel_check: CancelCheck | None = None,
        episode_sink: EpisodeSink | None = None,
    ) -> Any: ...

