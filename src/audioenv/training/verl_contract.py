"""Executable compatibility audit for the pinned VeRL integration surface."""

from __future__ import annotations

import importlib
import inspect
import json
from typing import Any, Callable


REQUIRED_OUTPUT_FIELDS = {
    "prompt_ids", "response_ids", "response_mask", "response_logprobs",
    "multi_modal_data", "reward_score", "num_turns", "metrics",
    "extra_fields", "mm_processor_kwargs",
}


def validate_verl_runtime_contract(
    *, importer: Callable[[str], Any] = importlib.import_module,
    object_loader: Callable[[str, str], Any] | None = None,
) -> dict[str, Any]:
    """Prove that installed VeRL exposes every API AudioEnv patches or calls."""

    agent_module = importer("verl.experimental.agent_loop.agent_loop")
    output = agent_module.AgentLoopOutput
    fields = set(getattr(output, "model_fields", {}))
    missing = sorted(REQUIRED_OUTPUT_FIELDS - fields)
    if missing:
        raise RuntimeError("VeRL AgentLoopOutput is missing: " + ", ".join(missing))
    manager = agent_module.AgentLoopManager
    if not inspect.iscoroutinefunction(manager._agent_loop_postprocess):
        raise RuntimeError("VeRL AgentLoopManager._agent_loop_postprocess must remain async")
    registry = agent_module._agent_loop_registry
    importer("audioenv.training.verl_plugin")
    registered = registry.get("audioenv_agent")
    if not isinstance(registered, dict) or "AudioEnvVerlAgentLoop" not in str(registered.get("_target_", "")):
        raise RuntimeError("AudioEnv agent loop did not register with pinned VeRL")

    if object_loader is None:
        object_loader = importer("verl.utils.import_utils").load_extern_object
    scorer = object_loader("pkg://audioenv.training.verl_plugin", "compute_audioenv_score")
    signature = inspect.signature(scorer)
    for name in ("data_source", "solution_str", "ground_truth", "extra_info"):
        if name not in signature.parameters:
            raise RuntimeError(f"AudioEnv reward scorer is missing parameter {name}")
    verifier_hash = "a" * 64
    reward = scorer(
        "audioenv", "ignored", None,
        {
            "audioenv_trusted_reward": {
                "terminal": 1.0, "process": 0.25,
                "reward_version": "v1", "verifier_hash": verifier_hash,
            },
            "audioenv_verifier_hash": verifier_hash,
            "audioenv_rollout_draft": {
                "task_id": "contract", "seed": 7, "audio_profile_id": "native-clean",
            },
        },
    )
    if not isinstance(reward, dict) or reward.get("score") != 1.25:
        raise RuntimeError("AudioEnv reward scorer did not preserve protected reward components")

    reward_registry = importer("verl.experimental.reward_loop.reward_manager")
    reward_manager = reward_registry.get_reward_manager_cls("naive")
    if not inspect.isclass(reward_manager):
        raise RuntimeError("VeRL naive reward manager is not registered")
    return {
        "schema_version": "audioenv.verl-contract/v1",
        "agent_loop": str(registered["_target_"]),
        "agent_output_fields": sorted(REQUIRED_OUTPUT_FIELDS),
        "reward_function": f"{scorer.__module__}.{scorer.__name__}",
        "reward_manager": f"{reward_manager.__module__}.{reward_manager.__name__}",
        "protected_reward_probe": reward,
    }


def main() -> int:
    print(json.dumps(validate_verl_runtime_contract(), sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
