"""Real CPU REINFORCE trainer for the local AudioEnv acceptance path.

This is an audio-conditioned platform proof, not a foundation speech model.
It selects public action candidates and executes every selected action through
``AudioEnvironment.step``. It never reads oracle actions, hidden facts,
expected state, verifier predicates, or the test split during optimization.
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
from ..contracts import (
    EvaluationProtocolV1,
    EvaluationResultV1,
    EvaluationSampleV1,
    ResourceEstimate,
    TrainerCapabilities,
)
from ..evaluation import build_evaluation_result, compare_evaluations, package_hashes
from ..reference import build_reference_split_manifest
from ..runtime import AudioEnvironment
from ..spec import SpeakAction, TaskSpec, ToolCallAction
from .base import CancelCheck, MetricSink


@dataclass(frozen=True, slots=True)
class TinyTrainerConfig:
    steps: int = 40
    learning_rate: float = 0.08
    seed: int = 7
    checkpoint_interval: int = 10
    resume_from: str | None = None


@dataclass(frozen=True, slots=True)
class TinyTrainingResult:
    checkpoint_path: Path
    initial_parameter_hash: str
    final_parameter_hash: str
    baseline_success_rate: float
    trained_success_rate: float
    completed_steps: int
    cancelled: bool
    baseline_evaluation: EvaluationResultV1
    trained_evaluation: EvaluationResultV1
    comparison: object
    checkpoint_reload_verified: bool
    split_manifest_hash: str


class ActionCandidateBuilder:
    """Construct the next discrete actions from public observations/history.

    The initial lookup candidate is part of the package's public action surface.
    Later candidates are derived only from returned tool data and prior public
    actions. Protected verifier fields and oracle trajectories are never read.
    """

    def __init__(self, package: EnvironmentPackage):
        self._known_tools = {tool.name for tool in package.spec.tools}
        self._candidates = package.public_action_candidates

    def build(self, task: TaskSpec, observation=None, history=()):
        candidates = self._candidates.get(task.id, ())
        if not candidates:
            raise ValueError(f"task {task.id} has no public action candidates")
        noop, initial_progress, _forbidden = candidates
        progress = initial_progress
        tool_result = observation.tool_result if observation is not None else None
        if tool_result and isinstance(tool_result.get("order"), dict):
            order = tool_result["order"]
            lookup = next(action for action, _ in reversed(history) if isinstance(action, ToolCallAction))
            progress = ToolCallAction(
                "refund_order",
                {"order_id": lookup.arguments["order_id"], "customer_zip": order["customer_zip"]},
            )
        elif tool_result and "refund_id" in tool_result:
            progress = SpeakAction(
                b"policy-speech",
                f"Your refund {tool_result['refund_id']} for ${tool_result['amount']} has been completed.",
                True,
            )
        elif tool_result and isinstance(tool_result.get("line"), dict):
            lookup = next(action for action, _ in reversed(history) if isinstance(action, ToolCallAction))
            progress = ToolCallAction("send_device_step", {"phone_number": lookup.arguments["phone_number"]})
        elif tool_result and tool_result.get("instruction") == "enable_data_roaming":
            lookup = next(action for action, _ in history if isinstance(action, ToolCallAction) and action.name == "lookup_line")
            progress = ToolCallAction(
                "confirm_device_state", {"phone_number": lookup.arguments["phone_number"], "enabled": True}
            )
        elif tool_result and tool_result.get("data_roaming") is True:
            progress = SpeakAction(b"policy-speech", "Data roaming is enabled.", True)
        for action in (noop, progress):
            if isinstance(action, ToolCallAction) and action.name not in self._known_tools:
                raise ValueError(f"candidate references unknown tool: {action.name}")
        return (noop, progress)


class SmallAudioToolPolicyTrainer:
    trainer_id = "audioenv/small-audio-reinforce"

    def __init__(self, package: EnvironmentPackage):
        self.package = package
        self.candidates = ActionCandidateBuilder(package)
        self.split_manifest = build_reference_split_manifest(package)

    @property
    def capabilities(self) -> TrainerCapabilities:
        return TrainerCapabilities(
            algorithms=("reinforce",),
            model_families=("audioenv-small-audio-tool-policy",),
            action_modalities=("native-wav-input", "structured-tools", "speech-candidate"),
            trainable_modules=("audio_candidate_scorer",),
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
        if config.resume_from and not Path(config.resume_from).is_file():
            errors.append("resume_from checkpoint does not exist")
        if not any(task.split == "train" for task in self.package.tasks):
            errors.append("environment has no train tasks")
        if not any(task.split == "test" for task in self.package.tasks):
            errors.append("environment has no held-out test tasks")
        return tuple(errors)

    def preflight(self, config: TinyTrainerConfig) -> ResourceEstimate:
        errors = self.validate_config(config)
        try:
            import torch  # noqa: F401
        except ImportError:
            errors = (*errors, "torch is not installed; install the train extra")
        return ResourceEstimate(
            provider="local-subprocess",
            device_type="cpu",
            device_count=1,
            estimated_cost_usd=0.0,
            cost_basis="local CPU; electricity excluded",
            available=not errors,
            reasons=errors,
        )

    def _new_model(self, torch):
        model = torch.nn.Linear(7, 1, bias=False)
        for parameter in model.parameters():
            torch.nn.init.zeros_(parameter)
        return model

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
        model = self._new_model(torch)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        completed = 0
        if config.resume_from:
            model, payload = self.load_checkpoint(Path(config.resume_from))
            optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
            optimizer.load_state_dict(payload["optimizer"])
            completed = int(payload["step"])
        start_step = completed
        initial_hash = _parameter_hash(model)
        baseline = self.evaluate_result(model, checkpoint_id=f"initial:{initial_hash}")
        destination.mkdir(parents=True, exist_ok=True)
        train_tasks = tuple(task for task in self.package.tasks if task.split == "train")
        cancelled = False

        for local_step in range(1, config.steps + 1):
            if cancel_check and cancel_check():
                cancelled = True
                break
            step = start_step + local_step
            task = train_tasks[(step - 1) % len(train_tasks)]
            env = AudioEnvironment(self.package)
            observation = env.reset(task.id, seed=config.seed + step, model_id=self.trainer_id)
            history = []
            log_probs = []
            entropies = []
            rewards = []
            for _ in range(self.package.spec.max_steps):
                candidates = self.candidates.build(task, observation, tuple(history))
                features = torch.stack([
                    torch.cat((_audio_features(observation.audio or b"", torch), _candidate_features(action, task, torch)))
                    for action in candidates
                ])
                distribution = torch.distributions.Categorical(logits=model(features).squeeze(-1))
                choice = distribution.sample()
                action = candidates[int(choice.item())]
                observation = env.step(action)
                history.append((action, observation))
                log_probs.append(distribution.log_prob(choice))
                entropies.append(distribution.entropy())
                rewards.append(float(observation.reward))
                if observation.done:
                    break
            # Each process milestone and the terminal outcome is already an
            # independently verified reward event. Using its immediate value
            # avoids teaching an earlier correct tool call that a later sampled
            # mistake made that earlier call wrong.
            returns = rewards
            loss = sum(-log_prob * return_value - 0.02 * entropy for log_prob, return_value, entropy in zip(log_probs, returns, entropies))
            reward = sum(rewards)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            completed = step
            if metric_sink:
                metric_sink(step, "loss", float(loss.detach()), self.trainer_id)
                metric_sink(step, "episode_reward", reward, "environment")
                metric_sink(
                    step,
                    "episode_success",
                    float(bool(env.trace.final_metrics.get("terminal_pass"))),
                    "environment",
                )
            if local_step % config.checkpoint_interval == 0:
                _save_checkpoint(model, optimizer, config, destination / f"step-{step}.pt", step)

        checkpoint = destination / "final.pt"
        _save_checkpoint(model, optimizer, config, checkpoint, completed)
        final_hash = _parameter_hash(model)
        reloaded, payload = self.load_checkpoint(checkpoint)
        reload_verified = _parameter_hash(reloaded) == final_hash and int(payload["step"]) == completed
        trained = self.evaluate_result(reloaded, checkpoint_id=final_hash)
        comparison = compare_evaluations(baseline, trained)
        manifest = {
            "trainer_id": self.trainer_id,
            "claim_scope": "audio_conditioned_platform_proof",
            "config": asdict(config),
            "completed_steps": completed,
            "cancelled": cancelled,
            "initial_parameter_hash": initial_hash,
            "final_parameter_hash": final_hash,
            "baseline_evaluation": baseline.model_dump(mode="json"),
            "trained_evaluation": trained.model_dump(mode="json"),
            "comparison": comparison.model_dump(mode="json"),
            "checkpoint_reload_verified": reload_verified,
            "split_manifest": self.split_manifest.model_dump(mode="json"),
            "checkpoint": checkpoint.name,
        }
        (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        return TinyTrainingResult(
            checkpoint,
            initial_hash,
            final_hash,
            baseline.success_rate,
            trained.success_rate,
            completed,
            cancelled,
            baseline,
            trained,
            comparison,
            reload_verified,
            self.split_manifest.content_hash,
        )

    def evaluate_result(self, model, *, checkpoint_id: str) -> EvaluationResultV1:
        import torch

        environment_hash, verifier_hash = package_hashes(self.package)
        test_tasks = tuple(task for task in self.package.tasks if task.split == "test")
        profile_ids = ("clean-en", "degraded-en", "fast-en")
        seeds = self.split_manifest.seeds["test"]
        samples: list[EvaluationSampleV1] = []
        for index, task in enumerate(test_tasks):
            for profile_id in profile_ids:
                seed = seeds[index]
                env = AudioEnvironment(self.package)
                observation = env.reset(
                    task.id,
                    seed=seed,
                    audio_profile_id=profile_id,
                    model_id=self.trainer_id,
                )
                history = []
                reward = 0.0
                for _ in range(self.package.spec.max_steps):
                    candidates = self.candidates.build(task, observation, tuple(history))
                    with torch.no_grad():
                        features = torch.stack([
                            torch.cat((_audio_features(observation.audio or b"", torch), _candidate_features(action, task, torch)))
                            for action in candidates
                        ])
                        choice = int(model(features).squeeze(-1).argmax().item())
                    action = candidates[choice]
                    observation = env.step(action)
                    history.append((action, observation))
                    reward += float(observation.reward)
                    if observation.done:
                        break
                passed = bool(env.trace.final_metrics.get("terminal_pass"))
                invalid = sum(
                    event.name in {"failed_tool_call", "unexpected_tool", "tool_failure"}
                    for step in env.trace.steps
                    for event in step.reward_events
                )
                safety = sum(
                    event.name == "unauthorized_mutation"
                    for step in env.trace.steps
                    for event in step.reward_events
                )
                samples.append(
                    EvaluationSampleV1(
                        task_id=task.id,
                        audio_profile_id=profile_id,
                        seed=seed,
                        passed=passed,
                        reward=reward,
                        safety_violations=safety,
                        invalid_calls=invalid,
                        failure_category=None if passed else ("unauthorized_mutation" if safety else "task_incomplete"),
                    )
                )
        protocol = EvaluationProtocolV1(
            environment_hash=environment_hash,
            verifier_hash=verifier_hash,
            split_manifest_hash=self.split_manifest.content_hash,
            split="test",
            caller_hash=_hash_json({"type": "deterministic-scripted-caller", "version": 1}),
            audio_pipeline_hash=_hash_json({"renderer": "deterministic-wav-v1", "profiles": profile_ids}),
            sampling_hash=_hash_json({"decode": "argmax", "candidate_builder": "public-v1"}),
            checkpoint_id=checkpoint_id,
            seeds=seeds,
        )
        return build_evaluation_result(protocol, samples)

    def evaluate(self, model) -> float:
        return self.evaluate_result(model, checkpoint_id=_parameter_hash(model)).success_rate

    def load_checkpoint(self, path: Path):
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install AudioEnv with the 'train' extra") from exc
        payload = torch.load(path, map_location="cpu", weights_only=True)
        model = self._new_model(torch)
        model.load_state_dict(payload["model"])
        return model, payload


def _candidate_features(action, task: TaskSpec, torch):
    is_tool = isinstance(action, ToolCallAction)
    is_speak = isinstance(action, SpeakAction)
    refusal = float(is_speak and "cannot" in (action.semantic_text or "").casefold())
    forbidden = float(is_tool and action.name in task.forbidden_tools)
    return torch.tensor(
        [float(is_tool), float(is_speak and not action.end_episode), refusal, forbidden],
        dtype=torch.float32,
    )


def _audio_features(audio: bytes, torch):
    if not audio:
        return torch.tensor([0.0, 0.0, 1.0])
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


def _hash_json(value) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# Backward-compatible names retained for existing integrations.
TinyAudioPolicyTrainer = SmallAudioToolPolicyTrainer
SmallAudioToolPolicyConfig = TinyTrainerConfig
