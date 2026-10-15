from audioenv.reference import build_reference_package
from audioenv.training import TinyAudioPolicyTrainer, TinyTrainerConfig


def test_tiny_trainer_updates_and_reloads(tmp_path):
    trainer = TinyAudioPolicyTrainer(build_reference_package())
    config = TinyTrainerConfig(steps=30, checkpoint_interval=10)
    assert trainer.preflight(config).available
    metrics = []
    result = trainer.train(config, tmp_path, metric_sink=lambda *args: metrics.append(args))
    assert result.initial_parameter_hash != result.final_parameter_hash
    assert result.trained_success_rate > result.baseline_success_rate
    model, payload = trainer.load_checkpoint(result.checkpoint_path)
    assert payload["step"] == 30
    assert trainer.evaluate(model) == result.trained_success_rate
    assert any(name == "loss" for _, name, _, _ in metrics)


def test_tiny_trainer_cancels_at_safe_boundary(tmp_path):
    trainer = TinyAudioPolicyTrainer(build_reference_package())
    result = trainer.train(
        TinyTrainerConfig(steps=20), tmp_path, cancel_check=lambda: True
    )
    assert result.cancelled
    assert result.completed_steps == 0
    assert result.checkpoint_path.exists()

