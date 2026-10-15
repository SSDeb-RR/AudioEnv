"""Small real policy-gradient trainer for CPU platform verification.

This validates orchestration and checkpoint behavior. It is deliberately not
presented as an omni-model or speech-quality training result.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import io
import json
from pathlib import Path
import random
import wave

from ..adapter import EnvironmentPackage
from ..contracts import ResourceEstimate, TrainerCapabilities
from ..runtime import AudioEnvironment
from ..spec import SpeakAction
from .base import CancelCheck, MetricSink


@dataclass(frozen=True, slots=True)
class TinyTrainerConfig:
    steps: int = 40
    learning_rate: float = 0.08
    seed: int = 7
    checkpoint_interval: int = 10


@dataclass(frozen=True, slots=True)
class TinyTrainingResult:
    checkpoint_path: Path
    initial_parameter_hash: str
    final_parameter_hash: str
    baseline_success_rate: float
    trained_success_rate: float
    completed_steps: int
    cancelled: bool


class TinyAudioPolicyTrainer:
    trainer_id = "audioenv/tiny-reinforce"

    def __init__(self, package: EnvironmentPackage):
        self.package = package

    @property
    def capabilities(self) -> TrainerCapabilities:
        return TrainerCapabilities(
            algorithms=("reinforce",),
            model_families=("audioenv-tiny-policy",),
            action_modalities=("audio-conditioned-strategy",),
            trainable_modules=("linear_policy",),
            supports_resume=True,
            checkpoint_format="pytorch-state-dict",
        )

    def validate_config(self, config: TinyTrainerConfig) -> tuple[str, ...]:
        errors: list[str] = []
        if config.steps < 1:
            errors.append("steps must be at least 1")
        if config.learning_rate <= 0:
            errors.append("learning_rate must be positive")
        if config.checkpoint_interval < 1:
            errors.append("checkpoint_interval must be at least 1")
        return tuple(errors)

    def preflight(self, config: TinyTrainerConfig) -> ResourceEstimate:
        errors = self.validate_config(config)
        try:
            import torch  # noqa: F401
        except ImportError:
            errors = (*errors, "torch is not installed; install the train extra")
        return ResourceEstimate(
            provider="local",
            device_type="cpu",
            device_count=1,
            available=not errors,
            reasons=errors,
        )

    def train(
        self,
        config: TinyTrainerConfig,
        destination: Path,
        *,
        metric_sink: MetricSink | None = None,
        cancel_check: CancelCheck | None = None,
    ) -> TinyTrainingResult:
        errors = self.validate_config(config)
        if errors:
            raise ValueError("; ".join(errors))
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install AudioEnv with the 'train' extra") from exc

        torch.manual_seed(config.seed)
        random.seed(config.seed)
        model = torch.nn.Linear(3, 2)
        with torch.no_grad():
            model.weight.zero_()
            model.bias[:] = torch.tensor([0.8, -0.8])
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        initial_hash = _parameter_hash(model)
        baseline = self.evaluate(model)
        destination.mkdir(parents=True, exist_ok=True)
        completed = 0
        cancelled = False

        for step in range(1, config.steps + 1):
            if cancel_check and cancel_check():
                cancelled = True
                break
            task = self.package.tasks[(step - 1) % len(self.package.tasks)]
            env = AudioEnvironment(self.package)
            observation = env.reset(task.id, seed=config.seed + step, model_id="audioenv-tiny-policy")
            features = _audio_features(observation.audio or b"", torch)
            distribution = torch.distributions.Categorical(logits=model(features))
            choice = distribution.sample()
            if int(choice.item()) == 1:
                for action in task.oracle_actions:
                    env.step(action)
            else:
                env.step(SpeakAction(b"tiny-noop", "I cannot complete this task.", True))
            if not env.public_state["done"]:
                env.step(SpeakAction(b"tiny-finish", "finished", True))
            reward = float(bool(env.trace.final_metrics.get("terminal_pass")))
            loss = -distribution.log_prob(choice) * reward - 0.01 * distribution.entropy()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            completed = step
            if metric_sink:
                metric_sink(step, "loss", float(loss.detach()), self.trainer_id)
                metric_sink(step, "episode_success", reward, "environment")
            if step % config.checkpoint_interval == 0:
                _save_checkpoint(model, optimizer, config, destination / f"step-{step}.pt", step)

        checkpoint = destination / "final.pt"
        _save_checkpoint(model, optimizer, config, checkpoint, completed)
        final_hash = _parameter_hash(model)
        trained = self.evaluate(model)
        manifest = {
            "trainer_id": self.trainer_id,
            "config": asdict(config),
            "completed_steps": completed,
            "cancelled": cancelled,
            "initial_parameter_hash": initial_hash,
            "final_parameter_hash": final_hash,
            "baseline_success_rate": baseline,
            "trained_success_rate": trained,
            "checkpoint": checkpoint.name,
        }
        (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        return TinyTrainingResult(checkpoint, initial_hash, final_hash, baseline, trained, completed, cancelled)

    def evaluate(self, model) -> float:
        import torch

        successes = 0
        for index, task in enumerate(self.package.tasks):
            env = AudioEnvironment(self.package)
            observation = env.reset(task.id, seed=10_000 + index, model_id="audioenv-tiny-policy")
            with torch.no_grad():
                choice = int(model(_audio_features(observation.audio or b"", torch)).argmax().item())
            if choice == 1:
                for action in task.oracle_actions:
                    env.step(action)
            else:
                env.step(SpeakAction(b"tiny-noop", "I cannot complete this task.", True))
            if not env.public_state["done"]:
                env.step(SpeakAction(b"tiny-finish", "finished", True))
            successes += int(bool(env.trace.final_metrics.get("terminal_pass")))
        return successes / len(self.package.tasks)

    @staticmethod
    def load_checkpoint(path: Path):
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install AudioEnv with the 'train' extra") from exc
        payload = torch.load(path, map_location="cpu", weights_only=True)
        model = torch.nn.Linear(3, 2)
        model.load_state_dict(payload["model"])
        return model, payload


def _audio_features(audio: bytes, torch):
    with wave.open(io.BytesIO(audio), "rb") as wav:
        frames = wav.readframes(wav.getnframes())
    samples = torch.frombuffer(bytearray(frames), dtype=torch.int16).float() / 32768.0
    if samples.numel() < 2:
        return torch.tensor([0.0, 0.0, 1.0])
    rms = torch.sqrt(torch.mean(samples.square()))
    crossings = torch.mean((samples[1:] * samples[:-1] < 0).float())
    return torch.stack((rms, crossings, torch.tensor(1.0)))


def _parameter_hash(model) -> str:
    digest = sha256()
    for parameter in model.parameters():
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _save_checkpoint(model, optimizer, config: TinyTrainerConfig, path: Path, step: int) -> None:
    import torch

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": asdict(config),
            "step": step,
        },
        path,
    )

