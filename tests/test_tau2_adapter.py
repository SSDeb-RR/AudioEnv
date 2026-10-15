import json

import audioenv.adapters.tau2 as tau2
from audioenv.adapters import Tau2RetailSourceAdapter


def test_tau2_source_import_preserves_official_split(tmp_path, monkeypatch):
    root = tmp_path / "tau2"
    domain = root / "data/tau2/domains/retail"
    domain.mkdir(parents=True)
    (root / "LICENSE").write_text("MIT")
    (domain / "tasks.json").write_text(json.dumps([{
        "id": "7",
        "user_scenario": {"instructions": {"reason_for_call": "refund", "known_info": "order 1"}},
        "evaluation_criteria": {"actions": [{"name": "refund", "arguments": {"id": "1"}}], "communicate_info": ["done"]},
    }]))
    (domain / "split_tasks.json").write_text(json.dumps({"train": [], "test": ["7"]}))
    monkeypatch.setattr(tau2, "_git_revision", lambda _: tau2.TAU2_V1_0_1_COMMIT)
    manifest = Tau2RetailSourceAdapter().import_source(root)
    assert manifest.source_revision == tau2.TAU2_V1_0_1_COMMIT
    assert manifest.tasks[0].split == "test"
    assert manifest.tasks[0].required_actions[0]["name"] == "refund"
