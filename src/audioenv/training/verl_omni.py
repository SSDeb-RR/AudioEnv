"""Validated launch boundary for VeRL-Omni Qwen3-Omni Thinker GSPO.

This module refuses to claim availability until the pinned checkout, CUDA
topology, model path, and launch recipe are present. It does not reimplement
VeRL-Omni's optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess

from ..contracts import ResourceEstimate, TrainerCapabilities


PINNED_VERL_OMNI_COMMIT = "eebe8a983b742960b0887026191517fe66505260"


@dataclass(frozen=True, slots=True)
class VerlOmniConfig:
    checkout: Path
    model_path: Path
    dataset_path: Path
    output_dir: Path
    nodes: int = 1
    gpus_per_node: int = 4
    learning_rate: float = 1e-6
    lora_rank: int = 32
    lora_alpha: int = 64
    max_steps: int = 1


class VerlOmniGSPOTrainer:
    trainer_id = "verl-omni/qwen3-thinker-gspo-lora"

    @property
    def capabilities(self) -> TrainerCapabilities:
        return TrainerCapabilities(
            algorithms=("gspo",),
            model_families=("qwen3-omni-thinker",),
            action_modalities=("audio-input-text-action",),
            trainable_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
            distributed_backends=("fsdp", "vllm-omni"),
            supports_resume=True,
            checkpoint_format="peft-lora",
        )

    def validate_config(self, config: VerlOmniConfig) -> tuple[str, ...]:
        errors: list[str] = []
        if config.gpus_per_node < 4:
            errors.append("the pinned validated recipe requires at least 4 GPUs")
        if config.nodes != 1:
            errors.append("V1 supports the pinned single-node recipe only")
        if config.lora_rank < 1 or config.lora_alpha < 1:
            errors.append("LoRA rank and alpha must be positive")
        if config.max_steps < 1:
            errors.append("max_steps must be positive")
        if not config.checkout.is_dir():
            errors.append("VeRL-Omni checkout does not exist")
        elif _git_revision(config.checkout) != PINNED_VERL_OMNI_COMMIT:
            errors.append(f"VeRL-Omni must be pinned to {PINNED_VERL_OMNI_COMMIT}")
        if not config.model_path.exists():
            errors.append("Qwen3-Omni model path does not exist")
        if not config.dataset_path.is_file():
            errors.append("trainer dataset does not exist")
        return tuple(errors)

    def preflight(self, config: VerlOmniConfig) -> ResourceEstimate:
        errors = list(self.validate_config(config))
        if shutil.which("nvidia-smi") is None:
            errors.append("nvidia-smi is unavailable")
        else:
            probe = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                check=False,
            )
            memories = [int(line.strip()) for line in probe.stdout.splitlines() if line.strip().isdigit()]
            if len(memories) < config.gpus_per_node:
                errors.append(f"requires {config.gpus_per_node} GPUs; found {len(memories)}")
            if any(value < 79_000 for value in memories[: config.gpus_per_node]):
                errors.append("each selected GPU must provide approximately 80 GB")
        return ResourceEstimate(
            provider="local-cuda",
            device_type="H800/H100-class CUDA GPU",
            device_count=config.gpus_per_node,
            memory_gb_per_device=80,
            available=not errors,
            reasons=tuple(errors),
        )

    def launch_command(self, config: VerlOmniConfig) -> tuple[str, ...]:
        preflight = self.preflight(config)
        if not preflight.available:
            raise RuntimeError("VeRL-Omni preflight failed: " + "; ".join(preflight.reasons))
        recipe = config.checkout / "examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_lora_avqa_v1.sh"
        if not recipe.is_file():
            raise RuntimeError("pinned VeRL-Omni recipe is missing")
        return ("bash", str(recipe))

    def launch_environment(self, config: VerlOmniConfig) -> dict[str, str]:
        env = {
            "MODEL_PATH": str(config.model_path.resolve()),
            "DATA_PATH": str(config.dataset_path.resolve()),
            "OUTPUT_DIR": str(config.output_dir.resolve()),
            "AUDIOENV_MAX_STEPS": str(config.max_steps),
            "AUDIOENV_LEARNING_RATE": str(config.learning_rate),
            "AUDIOENV_LORA_RANK": str(config.lora_rank),
            "AUDIOENV_LORA_ALPHA": str(config.lora_alpha),
        }
        return {**os.environ, **env}


def _git_revision(path: Path) -> str | None:
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True, check=False
    )
    return process.stdout.strip() if process.returncode == 0 else None

