from hashlib import sha256

from audioenv.artifacts import LocalArtifactStore
from audioenv.audio import DeterministicWavFixtureRenderer
from audioenv.contracts import GenerationRecord, ModelCapabilities, ProbeResult
from audioenv.reference import build_reference_package
from audioenv.rollout_v1 import V1RolloutRequest, V1RolloutWorker
from audioenv.spec import SpeakAction, ToolCallAction


class OracleV1Model:
    model_id = "test/oracle"

    def __init__(self, actions):
        self.actions = iter(actions)
        self.index = 0

    @property
    def capabilities(self):
        return ModelCapabilities(
            audio_input=True, audio_output=True, structured_tools=True,
            token_logprobs=True, trainable=False,
        )

    def probe(self):
        return ProbeResult(available=True, model_id=self.model_id, capabilities=self.capabilities)

    def generate_step(self, observation):
        action = next(self.actions)
        self.index += 1
        tool = None
        audio_id = None
        semantic = ""
        if isinstance(action, ToolCallAction):
            tool = {"tool_call": {"name": action.name, "arguments": action.arguments}}
        else:
            semantic = action.semantic_text or ""
            audio_id = sha256(action.audio).hexdigest()
        return action, GenerationRecord(
            generation_id=f"g{self.index}", model_id=self.model_id,
            model_revision="r1", policy_version="p1", semantic_text=semantic,
            input_ids=(1,), generated_ids=(2,), attention_mask=(1, 1),
            action_mask=(1,), old_logprobs=(-0.1,), audio_artifact_id=audio_id,
            tool_call=tool,
        )

    def save_adapter(self, destination):
        raise NotImplementedError

    def load_adapter(self, source):
        raise NotImplementedError


def test_v1_rollout_preserves_audio_and_training_alignment(tmp_path):
    package = build_reference_package()
    task = package.task("retail-refund-001")
    speech = task.oracle_actions[-1]
    audio = DeterministicWavFixtureRenderer().render("agent", package.spec.audio_profiles[0], 9)
    actions = (*task.oracle_actions[:-1], SpeakAction(audio, speech.semantic_text, True))
    worker = V1RolloutWorker(package, OracleV1Model(actions), LocalArtifactStore(tmp_path))
    trajectory = worker.run(V1RolloutRequest(
        experiment_id="e1", task_id=task.id, seed=3, checkpoint_id="base",
        policy_version="p1", processor_config={"sample_rate": 16_000},
    ))
    assert trajectory.terminal_score == 1.0
    assert trajectory.steps[-1].done
    assert len(trajectory.audio_artifacts) == 2
    assert trajectory.steps[-1].generation.audio_artifact_id in {
        item.artifact_id for item in trajectory.audio_artifacts
    }
