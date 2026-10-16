from pathlib import Path

from fastapi.testclient import TestClient

from audioenv.adapters.harbor import HarborEnvironmentAdapter
from audioenv.artifacts import LocalArtifactStore
from audioenv.integrations.harbor_sandbox import (
    DockerHarborTransport,
    HarborSandboxValidator,
    HarborTrialResult,
)
from audioenv.promotion import aggregate_validation_checks
from audioenv.platform.api import create_app


class FakeTransport:
    def __init__(self, *, reasons=()):
        self.reasons = reasons
        self.builds = []
        self.trials = []

    def probe(self):
        return self.reasons

    def build(self, *, task, image_tag, policy):
        self.builds.append((task.task_id, image_tag, policy))

    def trial(self, *, task, image_tag, mode, policy):
        del image_tag, policy
        self.trials.append((task.task_id, mode))
        if mode == "oracle":
            return HarborTrialResult(0, "a" * 64)
        if mode == "noop":
            return HarborTrialResult(1, "b" * 64)
        return HarborTrialResult(1, "c" * 64)


def _fixture(tmp_path: Path, *, adversarial: bool = True, dockerfile: str = "FROM scratch\n"):
    root = tmp_path / "tasks" / "fix-one"
    (root / "environment").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "solution").mkdir()
    if adversarial:
        (root / "adversarial").mkdir()
        (root / "adversarial" / "wrong.sh").write_text("#!/bin/sh\nexit 0\n")
    (root / "task.toml").write_text('id = "fix-one"\nsplit = "test"\n')
    (root / "instruction.md").write_text("Fix the public bug.")
    (root / "environment" / "Dockerfile").write_text(dockerfile)
    (root / "tests" / "test.sh").write_text("#!/bin/sh\n# SECRET_EXPECTATION\nexit 0\n")
    (root / "solution" / "solve.sh").write_text("#!/bin/sh\n# SECRET_SOLUTION\n")
    adapter = HarborEnvironmentAdapter(LocalArtifactStore(tmp_path / "artifacts"))
    return adapter.convert(root.parent, environment_id="fixture")


def test_harbor_validator_runs_fresh_negative_and_oracle_trials(tmp_path: Path):
    package, bundles = _fixture(tmp_path)
    transport = FakeTransport()
    report = HarborSandboxValidator(transport).validate(package, bundles)

    assert report["passed"] is True
    assert report["private_output_redacted"] is True
    assert transport.trials == [
        ("fix-one", "oracle"), ("fix-one", "oracle"),
        ("fix-one", "noop"), ("fix-one", "noop"),
        ("fix-one", "adversarial"),
    ]
    assert all(aggregate_validation_checks(report).values())
    encoded = str(report)
    assert "SECRET_EXPECTATION" not in encoded
    assert "SECRET_SOLUTION" not in encoded


def test_harbor_validator_fails_closed_without_adversarial_fixture(tmp_path: Path):
    package, bundles = _fixture(tmp_path, adversarial=False)
    report = HarborSandboxValidator(FakeTransport()).validate(package, bundles)
    assert report["passed"] is False
    assert aggregate_validation_checks(report)["adversarial_fail"] is False


def test_harbor_validator_reports_unavailable_sandbox_without_execution(tmp_path: Path):
    package, bundles = _fixture(tmp_path)
    transport = FakeTransport(reasons=("rootless Docker is required",))
    report = HarborSandboxValidator(transport).validate(package, bundles)
    assert report["passed"] is False
    assert report["checks"] == []
    assert transport.builds == []
    assert report["diagnostics"] == ["rootless Docker is required"]


def test_harbor_validator_rejects_mutable_or_secret_bearing_builds(tmp_path: Path):
    package, bundles = _fixture(tmp_path, dockerfile="FROM python:3.12\nRUN echo unsafe\n")
    transport = FakeTransport()

    mutable = HarborSandboxValidator(transport).validate(package, bundles)

    assert mutable["passed"] is False
    assert transport.builds == []
    assert "mutable base image" in str(mutable["checks"])

    package, bundles = _fixture(
        tmp_path / "secret", dockerfile=(
            "FROM registry.example/python@sha256:" + "a" * 64
            + "\nRUN --mount=type=secret,id=token echo unsafe\n"
        ),
    )
    secret_mount = HarborSandboxValidator(transport).validate(package, bundles)
    assert secret_mount["passed"] is False
    assert transport.builds == []
    assert "forbidden build mount" in str(secret_mount["checks"])


def test_harbor_validator_rejects_remote_add(tmp_path: Path):
    package, bundles = _fixture(
        tmp_path, dockerfile=(
            "FROM registry.example/base@sha256:" + "b" * 64
            + "\nADD https://example.com/payload /app/payload\n"
        ),
    )
    transport = FakeTransport()

    report = HarborSandboxValidator(transport).validate(package, bundles)

    assert report["passed"] is False
    assert transport.builds == []
    assert "remote ADD" in str(report["checks"])


def test_harbor_validator_rejects_source_mutation_after_import(tmp_path: Path):
    package, bundles = _fixture(tmp_path)
    transport = FakeTransport()
    (bundles[0].root / "environment" / "app.txt").write_text("changed after import")

    report = HarborSandboxValidator(transport).validate(package, bundles)

    assert report["passed"] is False
    assert transport.builds == []
    assert "source_immutability" in str(report["checks"])


def test_real_docker_probe_is_truthful():
    reasons = DockerHarborTransport().probe()
    assert isinstance(reasons, tuple)


def test_harbor_hash_binds_all_build_context_files(tmp_path: Path):
    first, _ = _fixture(tmp_path)
    extra = tmp_path / "tasks" / "fix-one" / "environment" / "app.txt"
    extra.write_text("version one")
    adapter = HarborEnvironmentAdapter(LocalArtifactStore(tmp_path / "second-artifacts"))
    second, _ = adapter.convert(tmp_path / "tasks", environment_id="fixture")
    assert first.content_hash != second.content_hash


def test_harbor_import_rejects_symlinks_anywhere_in_task(tmp_path: Path):
    _fixture(tmp_path)
    source = tmp_path / "outside.txt"
    source.write_text("outside")
    (tmp_path / "tasks" / "fix-one" / "environment" / "escape").symlink_to(source)
    adapter = HarborEnvironmentAdapter(LocalArtifactStore(tmp_path / "third-artifacts"))
    try:
        adapter.convert(tmp_path / "tasks", environment_id="fixture")
    except ValueError as exc:
        assert "symbolic links" in str(exc)
    else:
        raise AssertionError("Harbor symlink should be rejected")


def test_harbor_validation_api_persists_redacted_gate_evidence(tmp_path: Path):
    _fixture(tmp_path)
    database = f"sqlite:///{tmp_path / 'api.db'}"
    app = create_app(database, harbor_transport=FakeTransport())
    client = TestClient(app)
    created = client.post("/api/foundry/environments", json={
        "source_uri": str(tmp_path / "tasks"), "authoring_provider": "deterministic",
    })
    assert created.status_code == 201, created.text
    environment_id = created.json()["environment_id"]
    assert environment_id == "harbor/tasks@1.0.0"

    response = client.post(f"/api/environments/{environment_id}/validate-harbor", json={})
    assert response.status_code == 201, response.text
    payload = response.json()
    assert payload["validation"]["passed"] is True
    assert payload["validation"]["private_output_redacted"] is True
    assert payload["promotion"]["promoted"] is True
    assert payload["evidence_artifact_id"]
    details = client.get(f"/api/environments/{environment_id}").json()
    assert details["validation"]["passed"] is True
    assert details["authoring_status"] == "validated"
    assert "SECRET_EXPECTATION" not in response.text
