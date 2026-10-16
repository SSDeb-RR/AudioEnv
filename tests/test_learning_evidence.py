from audioenv.contracts import EvaluationProtocolV1, EvaluationSampleV1
from audioenv.evaluation import assess_learning_evidence, build_evaluation_result


def _protocol(checkpoint: str, *, caller: str = "c", audio: str = "a") -> EvaluationProtocolV1:
    return EvaluationProtocolV1(
        environment_hash="1" * 64,
        verifier_hash="2" * 64,
        split_manifest_hash="3" * 64,
        split="test",
        caller_hash=caller * 64,
        audio_pipeline_hash=audio * 64,
        sampling_hash="4" * 64,
        checkpoint_id=checkpoint,
        seeds=(11,),
    )


def _result(
    checkpoint: str,
    outcomes: tuple[bool, ...],
    *,
    caller: str = "c",
    audio: str = "a",
    safety: int = 0,
    repeats: int = 0,
):
    samples = tuple(
        EvaluationSampleV1(
            task_id=f"task-{index}", audio_profile_id="native", seed=11,
            passed=passed, reward=float(passed),
            safety_violations=safety if index == 0 else 0,
            repeated_failed_calls=repeats if index == 0 else 0,
        )
        for index, passed in enumerate(outcomes)
    )
    return build_evaluation_result(
        _protocol(checkpoint, caller=caller, audio=audio), samples,
    )


def test_learning_evidence_requires_primary_and_independent_transfer_gates():
    primary_baseline = _result("base", (False, False, True, True), safety=1, repeats=2)
    primary_trained = _result("trained", (True, False, True, True), safety=0, repeats=1)
    transfer_baseline = _result(
        "base", (False, False, True, True), caller="d", audio="b", safety=1, repeats=2,
    )
    transfer_trained = _result(
        "trained", (True, False, True, True), caller="d", audio="b", safety=0, repeats=1,
    )

    evidence = assess_learning_evidence(
        primary_baseline, primary_trained,
        independent_baseline=transfer_baseline,
        independent_trained=transfer_trained,
        reward_parity_verified=True,
        baseline_speech_quality=0.82,
        trained_speech_quality=0.80,
    )

    assert evidence.status == "validated"
    assert evidence.production_eligible is True
    assert evidence.primary_comparison.paired_delta == 0.25
    assert evidence.independent_comparison is not None
    assert all(check.status == "pass" for check in evidence.checks)


def test_learning_evidence_never_promotes_missing_external_evidence():
    baseline = _result("base", (False, False, True, True))
    trained = _result("trained", (True, False, True, True))

    evidence = assess_learning_evidence(baseline, trained)

    assert evidence.status == "incomplete"
    assert evidence.production_eligible is False
    assert {check.name for check in evidence.checks if check.status == "incomplete"} == {
        "independent_condition", "reward_parity", "speech_quality_non_regression",
    }


def test_learning_evidence_rejects_fake_independent_condition_and_regressions():
    baseline = _result("base", (False, False, True, True), safety=0, repeats=0)
    trained = _result("trained", (True, False, True, True), safety=1, repeats=1)

    evidence = assess_learning_evidence(
        baseline, trained,
        independent_baseline=_result("base", (False, False, True, True)),
        independent_trained=_result("trained", (True, False, True, True)),
        reward_parity_verified=False,
        baseline_speech_quality=0.9,
        trained_speech_quality=0.7,
    )

    assert evidence.status == "unmet"
    assert evidence.production_eligible is False
    failed = {check.name for check in evidence.checks if check.status == "fail"}
    assert {
        "independent_condition", "independent_learning_lift", "safety_non_regression",
        "repeated_call_non_regression", "reward_parity", "speech_quality_non_regression",
    } <= failed
