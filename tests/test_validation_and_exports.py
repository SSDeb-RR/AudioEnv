import json

from audioenv.export import export_package
from audioenv.integrations.verifiers import load_verifiers_taskset
from audioenv.reference import build_reference_package
from audioenv.validation import validate_package


def test_reference_package_validates():
    report = validate_package(build_reference_package())
    assert report.passed, [check for check in report.checks if not check.passed]


def test_export_redacts_oracles(tmp_path):
    path = export_package(build_reference_package(), tmp_path)
    text = path.read_text()
    assert "oracle_actions" not in text
    assert "hidden_facts" not in text
    assert "expected_state" not in text
    payload = json.loads(text)
    assert payload["format"] == "audioenv/v1"
    assert len(payload["tasks"]) == 2
    private = (tmp_path / ".audioenv-private" / "verifier.json").read_text()
    assert "expected_state" in private
    assert "oracle_actions" not in private


def test_verifiers_rows_do_not_leak_task_truth():
    rows = load_verifiers_taskset(build_reference_package())
    text = json.dumps(rows)
    assert "hidden_facts" not in text
    assert "expected_state" not in text
    assert "oracle" not in text

