import json

import pytest

from audioenv.artifacts import LocalArtifactStore
from audioenv.authoring import CustomTaskInput, DeclarativeTool, DeterministicEnvironmentAuthor
from audioenv.integrations.envharness import AudioEnvActionableBridge, RestrictedRulesAdapter, SetupAdapter
from audioenv.reference import build_reference_package
from audioenv.spec import SpeakAction, ToolCallAction


def test_custom_environment_is_executable_and_verifier_is_separate(tmp_path):
    source = CustomTaskInput(
        environment_id="custom/ticket",
        description="Close a ticket.",
        task_id="ticket-1",
        goal="Close ticket T1",
        initial_state={"ticket": {"status": "open"}},
        expected_state={"ticket": {"status": "closed"}},
        initial_utterance="Please close ticket T1.",
        tools=(
            DeclarativeTool(
                name="close_ticket",
                description="Close the ticket",
                operation="set_path",
                path=("ticket", "status"),
                required_arguments=("status",),
                value_argument="status",
            ),
        ),
        required_tools=(
            {"name": "close_ticket", "arguments": {"status": "closed"}, "milestone": "closed"},
        ),
        required_communications=("closed",),
    )
    authored = DeterministicEnvironmentAuthor(LocalArtifactStore(tmp_path)).author(source)
    assert authored.validation.passed
    public = authored.boundary.model_dump_json()
    assert '"expected_state"' not in public
    verifier = json.loads((tmp_path / authored.boundary.protected_verifier.sha256[:2] / authored.boundary.protected_verifier.sha256[2:]).read_text())
    assert verifier["expected_state"]["ticket"]["status"] == "closed"


def test_setup_and_rules_do_not_replace_evaluator():
    package = build_reference_package()
    task = package.task("retail-refund-001")
    bridge = AudioEnvActionableBridge(package, task.id)
    setup = SetupAdapter(bridge, (task.oracle_actions[0],))
    rules = RestrictedRulesAdapter(setup, frozenset({"lookup_order", "refund_order"}))
    response = rules.reset(3)
    assert not response.done
    rules.step(task.oracle_actions[1])
    rules.step(task.oracle_actions[2])
    assert rules.evaluate()["terminal_pass"] is True
    with pytest.raises(ValueError, match="blocked"):
        rules.step(ToolCallAction("change_plan", {"phone_number": "x", "plan": "x"}))


def test_official_envharness_contract_when_extra_is_installed():
    types = pytest.importorskip("envharness.core.types")
    from audioenv.integrations.envharness import OfficialAudioEnvBridge

    assert OfficialAudioEnvBridge is not None
    package = build_reference_package()
    task = package.task("retail-refund-001")
    bridge = OfficialAudioEnvBridge(package, task.id)
    reset = bridge.reset(seed=4)
    assert isinstance(reset, types.EnvResetResponse)
    for action in task.oracle_actions:
        if isinstance(action, ToolCallAction):
            external = types.Action(name=action.name, kwargs=action.arguments)
        else:
            external = types.Action(
                name="speak",
                kwargs={
                    "audio_hex": action.audio.hex(),
                    "semantic_text": action.semantic_text,
                    "end_episode": action.end_episode,
                },
            )
        response = bridge.step(external)
        assert isinstance(response, types.EnvResponse)
    result = bridge.evaluate()
    assert isinstance(result, types.EvaluationResult)
    assert result.success

