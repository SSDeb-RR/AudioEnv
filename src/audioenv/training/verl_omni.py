"""VeRL-Omni v0.2 Qwen3-Omni Thinker GSPO and GRPO launch plugins."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
from typing import Literal

from ..contracts import OnlineRLTrainerConfigV1, ResourceEstimate, TrainerCapabilities


PINNED_VERL_OMNI_COMMIT = "49dedb6ec0691588494b629fc1b610a86612966b"
PINNED_VERL_COMMIT = "8a694930275061f52ebd538c906ef8819af56dbd"


@dataclass(frozen=True, slots=True)
class VerlOmniConfig:
    checkout: Path
    model_path: Path
    dataset_path: Path
    validation_dataset_path: Path
    output_dir: Path
    environment_manifest: Path | None = None
    caller_model_path: Path | None = None
    algorithm: Literal["gspo", "grpo"] = "gspo"
    nodes: int = 1
    gpus_per_node: int = 4
    learning_rate: float = 1e-6
    lora_rank: int = 32
    lora_alpha: int = 64
    group_size: int = 4
    max_steps: int = 1
    evaluation_interval: int = 1
    evaluation_temperature: float = 0.0
    evaluation_top_p: float = 1.0
    evaluation_top_k: int = -1
    resume_from: Path | None = None

    def online(self) -> OnlineRLTrainerConfigV1:
        return OnlineRLTrainerConfigV1(
            algorithm=self.algorithm, model_id=str(self.model_path), group_size=self.group_size,
            learning_rate=self.learning_rate, lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha, max_steps=self.max_steps,
            resume_from=str(self.resume_from) if self.resume_from else None,
        )


class _VerlOmniTrainer:
    algorithm: Literal["gspo", "grpo"]

    @property
    def capabilities(self) -> TrainerCapabilities:
        return TrainerCapabilities(
            algorithms=(self.algorithm,), model_families=("qwen3-omni-thinker",),
            action_modalities=("native-audio-input", "structured-tools", "text-action-tokens"),
            trainable_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
            distributed_backends=("fsdp", "vllm-omni"), supports_resume=True,
            checkpoint_format="peft-lora",
        )

    def validate_config(self, config: VerlOmniConfig) -> tuple[str, ...]:
        errors: list[str] = []
        try:
            online = config.online()
        except Exception as exc:
            errors.append(str(exc))
            online = None
        if config.algorithm != self.algorithm:
            errors.append(f"{self.trainer_id} requires algorithm={self.algorithm}")
        if config.gpus_per_node < 4:
            errors.append("Qwen3-Omni Thinker LoRA requires at least 4 80GB GPUs")
        if config.nodes != 1:
            errors.append("the release-certified path is single-node")
        if config.evaluation_interval < 1:
            errors.append("evaluation_interval must be positive")
        if not math.isfinite(config.evaluation_temperature) or config.evaluation_temperature < 0:
            errors.append("evaluation_temperature must be finite and non-negative")
        if not math.isfinite(config.evaluation_top_p) or not 0 < config.evaluation_top_p <= 1:
            errors.append("evaluation_top_p must be finite and in (0, 1]")
        if config.evaluation_top_k < -1:
            errors.append("evaluation_top_k must be -1 or non-negative")
        if not config.checkout.is_dir():
            errors.append("VeRL-Omni checkout does not exist")
        elif _git_revision(config.checkout) != PINNED_VERL_OMNI_COMMIT:
            errors.append(f"VeRL-Omni must be pinned to {PINNED_VERL_OMNI_COMMIT}")
        if not config.model_path.exists():
            errors.append("Qwen3-Omni model path does not exist")
        if not config.dataset_path.is_file():
            errors.append("rollout seed dataset does not exist")
        if not config.validation_dataset_path.is_file():
            errors.append("held-out validation dataset does not exist")
        elif config.validation_dataset_path.resolve() == config.dataset_path.resolve():
            errors.append("training and held-out validation datasets must be different files")
        if config.environment_manifest is not None and not config.environment_manifest.is_file():
            errors.append("AudioEnv environment manifest does not exist")
        if config.caller_model_path is not None and not config.caller_model_path.exists():
            errors.append("frozen caller model path does not exist")
        if config.resume_from is not None and not config.resume_from.exists():
            errors.append("resume checkpoint does not exist")
        if online and tuple(online.lora_targets) != ("q_proj", "k_proj", "v_proj", "o_proj"):
            errors.append("only the release-certified Thinker LoRA targets are supported")
        return tuple(errors)

    def preflight(self, config: VerlOmniConfig) -> ResourceEstimate:
        errors = list(self.validate_config(config))
        if shutil.which("nvidia-smi") is None:
            errors.append("nvidia-smi is unavailable")
        else:
            process = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=False,
            )
            memory = [int(item.strip()) for item in process.stdout.splitlines() if item.strip().isdigit()]
            if len(memory) < config.gpus_per_node:
                errors.append(f"requires {config.gpus_per_node} GPUs; found {len(memory)}")
            if any(item < 79_000 for item in memory[:config.gpus_per_node]):
                errors.append("selected GPUs must provide approximately 80 GB each")
        return ResourceEstimate(
            provider="local-cuda", device_type="H800/H100/H200-class CUDA GPU",
            device_count=config.gpus_per_node, memory_gb_per_device=80,
            available=not errors, reasons=tuple(errors),
        )

    def launch_command(self, config: VerlOmniConfig, *, require_preflight: bool = True) -> tuple[str, ...]:
        errors = self.validate_config(config)
        if require_preflight:
            estimate = self.preflight(config)
            if not estimate.available:
                raise RuntimeError("VeRL-Omni preflight failed: " + "; ".join(estimate.reasons))
        elif errors:
            raise RuntimeError("invalid VeRL-Omni configuration: " + "; ".join(errors))
        loss_mode = "gspo" if self.algorithm == "gspo" else "vanilla"
        command = [
            "python3", "-m", "verl_omni.trainer.main_omni",
            f"data.train_files={config.dataset_path.resolve()}",
            f"data.val_files={config.validation_dataset_path.resolve()}",
            "data.truncation=error",
            "data.filter_overlong_prompts=true",
            f"actor_rollout_ref.model.path={config.model_path.resolve()}",
            f"actor_rollout_ref.model.lora_rank={config.lora_rank}",
            f"actor_rollout_ref.model.lora_alpha={config.lora_alpha}",
            "actor_rollout_ref.model.lora_dtype=float32",
            "actor_rollout_ref.model.lora.merge=True",
            "actor_rollout_ref.model.enable_gradient_checkpointing=True",
            "actor_rollout_ref.model.use_remove_padding=True",
            "actor_rollout_ref.model.exclude_modules=.*talker.*|.*code2wav.*|.*code_predictor.*|.*visual.*|.*audio_tower.*",
            "actor_rollout_ref.model.target_modules=['q_proj','k_proj','v_proj','o_proj']",
            "actor_rollout_ref.actor.freeze_vision_tower=true",
            "actor_rollout_ref.actor.strategy=fsdp2",
            f"actor_rollout_ref.actor.optim.lr={config.learning_rate}",
            "actor_rollout_ref.actor.use_kl_loss=false",
            f"actor_rollout_ref.actor.policy_loss.loss_mode={loss_mode}",
            "actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean",
            "actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16",
            f"actor_rollout_ref.rollout.n={config.group_size}",
            "actor_rollout_ref.rollout.name=vllm_omni",
            "actor_rollout_ref.rollout.tensor_model_parallel_size=2",
            "actor_rollout_ref.rollout.calculate_log_probs=True",
            "actor_rollout_ref.rollout.enable_prefix_caching=False",
            "actor_rollout_ref.rollout.val_kwargs.n=1",
            "actor_rollout_ref.rollout.val_kwargs.do_sample=false",
            f"actor_rollout_ref.rollout.val_kwargs.temperature={config.evaluation_temperature}",
            f"actor_rollout_ref.rollout.val_kwargs.top_p={config.evaluation_top_p}",
            f"actor_rollout_ref.rollout.val_kwargs.top_k={config.evaluation_top_k}",
            "+actor_rollout_ref.rollout.engine_kwargs.vllm_omni.output_mode=ar",
            "+actor_rollout_ref.rollout.engine_kwargs.vllm_omni.pipeline_name=qwen3_omni_moe",
            "+actor_rollout_ref.rollout.engine_kwargs.vllm_omni.pipeline_mode=full",
            "actor_rollout_ref.rollout.agent.default_agent_loop=audioenv_agent",
            "actor_rollout_ref.ref.strategy=fsdp2",
            "actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16",
            "algorithm.adv_estimator=grpo",
            "algorithm.use_kl_in_reward=False",
            "reward.reward_manager.source=register",
            "reward.reward_manager.name=naive",
            "reward.custom_reward_function.path=pkg://audioenv.training.verl_plugin",
            "reward.custom_reward_function.name=compute_audioenv_score",
            f"trainer.n_gpus_per_node={config.gpus_per_node}", f"trainer.nnodes={config.nodes}",
            f"trainer.total_training_steps={config.max_steps}",
            f"trainer.default_local_dir={config.output_dir.resolve()}",
            "trainer.val_before_train=true",
            f"trainer.test_freq={config.evaluation_interval}",
            "trainer.save_freq=1",
            f"trainer.validation_data_dir={(config.output_dir / 'validation' / 'training').resolve()}",
        ]
        if config.resume_from:
            command.extend(("trainer.resume_mode=resume_path", f"trainer.resume_from_path={config.resume_from.resolve()}"))
        return tuple(command)

    def reload_evaluation_command(self, config: VerlOmniConfig, checkpoint: Path) -> tuple[str, ...]:
        """Reload one saved checkpoint and run held-out validation only."""

        if not checkpoint.is_dir() or not checkpoint.name.startswith("global_step_"):
            raise ValueError("reload evaluation requires a global_step_ checkpoint directory")
        evaluation = replace(config, resume_from=checkpoint)
        command = list(self.launch_command(evaluation, require_preflight=False))
        replacements = {
            "trainer.val_before_train": "true",
            "trainer.test_freq": "-1",
            "trainer.validation_data_dir": str((config.output_dir / "validation" / "reloaded").resolve()),
        }
        for key, value in replacements.items():
            prefix = key + "="
            command = [item for item in command if not item.startswith(prefix)]
            command.append(prefix + value)
        command.append("trainer.val_only=true")
        return tuple(command)

    def launch_environment(self, config: VerlOmniConfig) -> dict[str, str]:
        plugin = {
            "schema_version": "audioenv.verl-plugin/v1", "algorithm": self.algorithm,
            "environment_manifest": str(config.environment_manifest.resolve()) if config.environment_manifest else None,
            "caller_model_path": str(config.caller_model_path.resolve()) if config.caller_model_path else None,
        }
        return {
            **os.environ, "AUDIOENV_VERL_PLUGIN": json.dumps(plugin, sort_keys=True),
            "AUDIOENV_VERL_ROLLOUT_FACTORY": "audioenv.training.verl_runtime:run_audioenv_trajectory",
            "AUDIOENV_NATIVE_RUNTIME_FACTORY": "audioenv.training.qwen_tau2_runtime:run_qwen_tau2_rollout",
            "AUDIOENV_ENABLE_NATIVE_AUDIO_BRIDGE": "1",
            # VeRL imports these modules in the driver and Ray worker processes.
            # Importing AudioEnv here makes its registered integration boundaries
            # available without importing VeRL in the local control plane.
            "VERL_USE_EXTERNAL_MODULES": "verl_omni,audioenv.training.verl_plugin",
        }


class VerlOmniGSPOTrainer(_VerlOmniTrainer):
    trainer_id = "qwen3-omni-thinker-gspo-lora"
    algorithm = "gspo"


class VerlOmniGRPOTrainer(_VerlOmniTrainer):
    trainer_id = "qwen3-omni-thinker-grpo-lora"
    algorithm = "grpo"


def _git_revision(path: Path) -> str | None:
    process = subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True, check=False)
    return process.stdout.strip() if process.returncode == 0 else None
