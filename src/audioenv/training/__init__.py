"""Trainer plugins and the local verifiable training path."""

from .base import TrainerPlugin
from .tiny import TinyAudioPolicyTrainer, TinyTrainerConfig, TinyTrainingResult
from .verl_omni import VerlOmniConfig, VerlOmniGSPOTrainer

__all__ = [
    "TrainerPlugin",
    "TinyAudioPolicyTrainer",
    "TinyTrainerConfig",
    "TinyTrainingResult",
    "VerlOmniConfig",
    "VerlOmniGSPOTrainer",
]

