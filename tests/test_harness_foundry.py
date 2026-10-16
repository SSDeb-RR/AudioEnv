import json
import io
import zipfile
import subprocess

import pytest
from fastapi.testclient import TestClient

from audioenv.artifacts import LocalArtifactStore
from audioenv.authoring import CustomTaskInput, DeclarativeTool
from audioenv.foundry import EnvironmentFoundry, ModelFoundry
from audioenv.harnesses import HarnessDescriptor, HarnessRegistry, build_builtin_registry, resolve_compatibility
from audioenv.harnesses.codex_author import CodexEnvironmentAuthor, build_source_digest
from audioenv.platform.api import create_app


def _task_manifest(path):
    task = CustomTaskInput(
        environment_id="foundry/tickets",
        description="Close a support ticket",
        task_id="ticket-1",
        goal="Close ticket T1",
        initial_state={"ticket": {"status": "open"}},
        expected_state={"ticket": {"status": "closed"}},
        initial_utterance="Please close ticket T1.",
        tools=(DeclarativeTool(
            name="close_ticket", description="Close ticket", operation="set_path",
            path=("ticket", "status"), required_arguments=("status",), value_argument="status",
        ),),
        required_tools=({"name": "close_ticket", "arguments": {"status": "closed"}, "milestone": "closed"},),
        required_communications=("closed",),
    )
    path.write_text(task.model_dump_json(), encoding="utf-8")


def test_registry_rejects_duplicates_and_resolves_real_compatibility():
    registry = build_builtin_registry()
    with pytest.raises(ValueError, match="duplicate"):
        registry.register(HarnessDescriptor(
            id="audioenv/small-audio-reinforce", kind="trainer", display_name="duplicate",
            status="available", status_reason="test",
        ))
    report = resolve_compatibility(
        registry,
        environment_id="env",
        environment_requirements=("audio-input", "structured-tool-calls"),
        model_id="model",
        model_capabilities=("native-wav-input", "structured-tools", "trainable", "pytorch-state-dict"),
    )
    local = next(item for item in report.trainers if item.harness_id == "audioenv/small-audio-reinforce")
    gpu = next(item for item in report.trainers if item.harness_id == "qwen3-omni-thinker-gspo-lora")
    assert local.compatible
    assert not gpu.compatible
    assert any("token-logprobs" in reason for reason in gpu.reasons)


def test_environment_foundry_compiles_declarative_repository_without_executing_code(tmp_path):
    repository = tmp_path / "source"
    repository.mkdir()
    _task_manifest(repository / "audioenv-task.json")
    (repository / "dangerous.py").write_text("raise RuntimeError('must never execute')\n")
    foundry = EnvironmentFoundry(tmp_path / "work", LocalArtifactStore(tmp_path / "artifacts"))
    result = foundry.build(str(repository))
    assert result.report.status == "available"
    assert result.report.classification == "declarative_audio_task"
    assert result.authored is not None and result.authored.validation.passed
    assert result.report.environment_id == "foundry/tickets@1.0.0"


def test_repository_tests_generate_blocked_sandbox_scaffold(tmp_path):
    repository = tmp_path / "code"
    repository.mkdir()
    (repository / "test_feature.py").write_text("def test_feature(): assert True\n")
    foundry = EnvironmentFoundry(tmp_path / "work", LocalArtifactStore(tmp_path / "artifacts"))
    result = foundry.build(str(repository))
    assert result.report.classification == "coding_repository"
    assert result.report.status == "needs_review"
    assert result.report.execution_mode == "oci_sandbox"
    assert result.generated_payload["sandbox"]["status"] == "required_not_configured"


def test_codex_authoring_is_schema_constrained_redacted_and_review_gated(tmp_path, monkeypatch):
    repository = tmp_path / "source"
    repository.mkdir()
    (repository / "README.md").write_text(
        "Create a ticket closer. api_key=do-not-leak", encoding="utf-8",
    )
    candidate_path = tmp_path / "candidate-source.json"
    _task_manifest(candidate_path)
    candidate = candidate_path.read_text(encoding="utf-8")
    observed = {}

    def runner(command, **kwargs):
        observed["command"] = command
        observed["prompt"] = kwargs["input"]
        output = command[command.index("--output-last-message") + 1]
        with open(output, "w", encoding="utf-8") as handle:
            handle.write(candidate)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("audioenv.harnesses.codex_author.shutil.which", lambda _: "/usr/bin/codex")
    author = CodexEnvironmentAuthor(runner=runner)
    foundry = EnvironmentFoundry(
        tmp_path / "work", LocalArtifactStore(tmp_path / "artifacts"), codex_author=author,
    )
    result = foundry.build(str(repository), authoring_provider="codex")
    assert result.report.classification == "codex_generated_candidate"
    assert result.report.status == "needs_review"
    assert result.authored is not None
    assert result.authored.boundary.authoring_status == "generated_needs_review"
    assert "do-not-leak" not in observed["prompt"]
    assert "[REDACTED]" in observed["prompt"]
    assert "read-only" in observed["command"]
    assert "--output-schema" in observed["command"]


def test_source_digest_skips_symlinks_and_secret_files(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    (root / "README.md").write_text("password=hunter2", encoding="utf-8")
    (root / ".env").write_text("TOKEN=secret", encoding="utf-8")
    (root / "linked.md").symlink_to(root / "README.md")
    digest = build_source_digest(root, ("README.md", ".env", "linked.md"))
    assert "hunter2" not in digest
    assert "TOKEN=secret" not in digest
    assert digest.count("README.md") == 1


def test_model_foundry_selects_family_driver_and_scaffolds_unknown(tmp_path):
    registry = build_builtin_registry()
    foundry = ModelFoundry(registry, LocalArtifactStore(tmp_path / "artifacts"))
    qwen = foundry.inspect("Qwen/Qwen3-Omni-30B-A3B-Instruct", "Qwen/Qwen3-Omni-30B-A3B-Instruct")
    assert qwen.detected_family == "qwen3-omni"
    assert qwen.selected_driver == "audioenv/qwen3-omni"
    assert qwen.status == "blocked"
    unknown_dir = tmp_path / "unknown"
    unknown_dir.mkdir()
    (unknown_dir / "config.json").write_text(json.dumps({"model_type": "new_audio_arch"}))
    unknown = foundry.inspect("lab/new-audio", str(unknown_dir))
    assert unknown.status == "needs_review"
    assert unknown.generated_files
    archive = zipfile.ZipFile(io.BytesIO(foundry.artifacts.get_bytes(unknown.generated_files[0])))
    assert "manifest.json" in archive.namelist()
    manifest = json.loads(archive.read("manifest.json"))
    assert manifest["registration_state"] == "generated_needs_review"
    assert "lora_update" in manifest["required_conformance"]
    assert "adapter_plan_sha256" in manifest
    adapter_plan = json.loads(archive.read("adapter_spec.json"))
    assert adapter_plan["schema_version"] == "audioenv.omni-adapter-plan/v1"
    assert adapter_plan["implementation_order"][-1].startswith("pass every isolated")
    driver_source = archive.read(next(name for name in archive.namelist() if name.endswith("/driver.py"))).decode()
    assert "conformance_optimizer_step" in driver_source
    assert "NotImplementedError" in driver_source


def test_foundry_api_persists_build_and_drives_compatibility(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _task_manifest(source / "audioenv-task.json")
    client = TestClient(create_app(f"sqlite:///{tmp_path / 'foundry.sqlite'}"))
    project = client.post("/api/projects", json={"name": "Foundry", "description": "dynamic"}).json()
    build = client.post("/api/foundry/environments", json={
        "project_id": project["id"], "source_uri": str(source),
    })
    assert build.status_code == 201, build.text
    payload = build.json()
    assert payload["status"] == "available"
    assert payload["environment_id"] == "foundry/tickets@1.0.0"
    listed = client.get("/api/foundry/builds?kind=environment").json()
    assert listed[0]["id"] == payload["id"]
    reference = next(item for item in client.get("/api/environments").json() if item["source_kind"] == "builtin")
    model = next(item for item in client.get("/api/models").json() if item["available"])
    compatibility = client.post("/api/compatibility", json={
        "environment_id": reference["id"], "model_id": model["id"],
    })
    assert compatibility.status_code == 200, compatibility.text
    local = next(item for item in compatibility.json()["trainers"] if item["harness_id"] == "audioenv/small-audio-reinforce")
    assert local["compatible"]


def test_unknown_model_api_registers_downloadable_review_gated_sdk(tmp_path):
    source = tmp_path / "unknown-model"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({"model_type": "novel_audio"}))
    app = create_app(f"sqlite:///{tmp_path / 'models.sqlite'}")
    client = TestClient(app)
    project = client.post("/api/projects", json={"name": "Adapters"}).json()
    response = client.post("/api/foundry/models", json={
        "project_id": project["id"], "model_id": "lab/novel-audio", "source": str(source),
    })
    assert response.status_code == 201
    artifact_id = response.json()["generated_artifact_ids"][0]
    artifact = app.state.repository.get_artifact(artifact_id)
    assert artifact.kind == "model-driver-scaffold"
    assert artifact.media_type == "application/zip"
    assert client.get(f"/api/artifacts/{artifact_id}/download").content.startswith(b"PK\x03\x04")
    model = app.state.repository.get_model("lab/novel-audio")
    assert model.available is False
    assert model.adapter_type == "generated-needs-review"
    assert model.registration_state == "generated_needs_review"
