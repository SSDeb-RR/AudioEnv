import pytest

from audioenv.adapters import JsonBenchmarkAdapter
from audioenv.reference import build_reference_package


def test_exported_public_package_can_be_inspected(tmp_path):
    package = build_reference_package()
    source = _authoring_manifest(tmp_path, package)
    adapter = JsonBenchmarkAdapter(package.tool_handlers)
    summary = adapter.inspect(source)
    assert summary["task_count"] == 48
    assert summary["splits"] == ["test", "train", "validation"]


def test_conversion_requires_explicit_audited_handlers(tmp_path):
    package = build_reference_package()
    source = _authoring_manifest(tmp_path, package)
    with pytest.raises(ValueError, match="missing audited tool handlers"):
        JsonBenchmarkAdapter({}).convert(source)


def test_authoring_manifest_imports_oracle(tmp_path):
    package = build_reference_package()
    source = _authoring_manifest(tmp_path, package)
    converted = JsonBenchmarkAdapter(package.tool_handlers).convert(source)
    assert converted.tasks[0].oracle_actions[0].name == "lookup_order"


def _authoring_manifest(tmp_path, package):
    import json
    from dataclasses import asdict

    payload = {
        "format": "audioenv/v1",
        "environment": asdict(package.spec),
        "tasks": [],
    }
    for task in package.tasks:
        raw = asdict(task)
        raw["oracle_actions"] = [
            {
                **asdict(action),
                **({"audio_base64": ""} if action.kind == "speak" else {}),
            }
            for action in task.oracle_actions
        ]
        for action in raw["oracle_actions"]:
            action.pop("audio", None)
        payload["tasks"].append(raw)
    source = tmp_path / "authoring.json"
    source.write_text(json.dumps(payload))
    return source

