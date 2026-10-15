from audioenv.evaluation import EvaluationSample, package_hashes, summarize
from audioenv.reference import build_reference_package


def test_evaluation_summary_and_hashes_are_deterministic():
    samples = [
        EvaluationSample("a", 1, True, 1.0),
        EvaluationSample("b", 2, False, 0.0, invalid_calls=1),
    ]
    summary = summarize(samples, bootstrap_rounds=100)
    assert summary.success_rate == 0.5
    assert summary.invalid_calls == 1
    first = package_hashes(build_reference_package())
    second = package_hashes(build_reference_package())
    assert first == second
    assert all(len(value) == 64 for value in first)
