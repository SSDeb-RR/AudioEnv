from audioenv.reference import build_reference_package
from audioenv.runtime import AudioEnvironment


def test_public_trace_redacts_semantic_sidecar():
    package = build_reference_package()
    task = package.task("retail-refund-001")
    env = AudioEnvironment(package)
    env.reset(task.id)
    for action in task.oracle_actions:
        env.step(action)
    public = env.trace.to_json()
    private = env.trace.to_json(include_observability=True)
    assert "Your refund" not in public
    assert "Your refund" in private


def test_deterministic_audio_for_same_seed():
    package = build_reference_package()
    one = AudioEnvironment(package).reset("retail-refund-001", seed=123)
    two = AudioEnvironment(package).reset("retail-refund-001", seed=123)
    other = AudioEnvironment(package).reset("retail-refund-001", seed=124)
    assert one.audio == two.audio
    assert one.audio != other.audio

