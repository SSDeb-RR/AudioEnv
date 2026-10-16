from types import SimpleNamespace

import pytest

from audioenv.training.verl_contract import REQUIRED_OUTPUT_FIELDS, validate_verl_runtime_contract
from audioenv.training.verl_plugin import compute_audioenv_score


class _Output:
    model_fields = {name: object() for name in REQUIRED_OUTPUT_FIELDS}


class _Manager:
    async def _agent_loop_postprocess(self):
        return None


class _QueueWorker:
    async def _agent_loop_postprocess(self):
        return None


class _RewardManager:
    pass


class _NativeServer:
    pass


class _FSDPEngine:
    pass


class _PPOTrainer:
    pass


def _runtime(*, missing: str | None = None, register: bool = True, bridges: bool = False):
    fields = dict(_Output.model_fields)
    if missing:
        fields.pop(missing)
    output = type("Output", (), {"model_fields": fields})
    registry = {}
    agent = SimpleNamespace(
        AgentLoopOutput=output, AgentLoopManager=_Manager,
        _agent_loop_registry=registry,
    )
    rewards = SimpleNamespace(get_reward_manager_cls=lambda name: _RewardManager if name == "naive" else None)

    def importer(name):
        if name == "verl.experimental.agent_loop.agent_loop":
            return agent
        if name == "audioenv.training.verl_plugin":
            if register:
                registry["audioenv_agent"] = {"_target_": "audioenv.training.verl_plugin.AudioEnvVerlAgentLoop"}
            if bridges:
                _QueueWorker._audioenv_trajectory_bridge = True
                _NativeServer._audioenv_native_bridge = True
                _FSDPEngine._audioenv_gradient_audit = True
                _PPOTrainer._audioenv_logprob_audit = True
            return SimpleNamespace()
        if name == "verl.experimental.reward_loop.reward_manager":
            return rewards
        if name == "verl.trainer.ppo.v1.agent_loop_tq":
            return SimpleNamespace(AgentLoopWorkerTQ=_QueueWorker)
        if name == "verl_omni.workers.rollout.vllm_rollout.vllm_omni_async_server":
            return SimpleNamespace(VLLMOmniAsyncServer=_NativeServer)
        if name == "verl.workers.engine.fsdp.transformer_impl":
            return SimpleNamespace(FSDPEngine=_FSDPEngine)
        if name == "verl.trainer.ppo.v1.trainer_base":
            return SimpleNamespace(PPOTrainer=_PPOTrainer)
        raise AssertionError(name)

    return importer


def test_verl_contract_accepts_agent_multimodal_and_protected_reward_surface():
    report = validate_verl_runtime_contract(
        importer=_runtime(), object_loader=lambda _path, _name: compute_audioenv_score,
    )
    assert report["schema_version"] == "audioenv.verl-contract/v1"
    assert report["protected_reward_probe"]["score"] == 1.25
    assert "AudioEnvVerlAgentLoop" in report["agent_loop"]


def test_verl_contract_fails_on_upstream_field_or_registration_drift():
    with pytest.raises(RuntimeError, match="mm_processor_kwargs"):
        validate_verl_runtime_contract(
            importer=_runtime(missing="mm_processor_kwargs"),
            object_loader=lambda _path, _name: compute_audioenv_score,
        )
    with pytest.raises(RuntimeError, match="did not register"):
        validate_verl_runtime_contract(
            importer=_runtime(register=False),
            object_loader=lambda _path, _name: compute_audioenv_score,
        )


def test_verl_contract_requires_all_training_runtime_bridges_when_requested():
    report = validate_verl_runtime_contract(
        importer=_runtime(bridges=True),
        object_loader=lambda _path, _name: compute_audioenv_score,
        require_runtime_bridges=True,
    )
    assert all(report["runtime_bridges"].values())
    _PPOTrainer._audioenv_logprob_audit = False
    with pytest.raises(RuntimeError, match="logprob_parity_audit"):
        validate_verl_runtime_contract(
            importer=_runtime(bridges=False),
            object_loader=lambda _path, _name: compute_audioenv_score,
            require_runtime_bridges=True,
        )
