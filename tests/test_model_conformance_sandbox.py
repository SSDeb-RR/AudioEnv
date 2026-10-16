import json
from pathlib import Path

from fastapi.testclient import TestClient

from audioenv.contracts import ConformanceCheckV1, ModelConformanceReportV1
from audioenv.integrations.model_conformance_sandbox import ModelConformanceSandbox
from audioenv.integrations.model_conformance_sandbox import (
    DockerModelConformanceImageBuilder, DockerModelConformanceTransport,
)
from audioenv.models.conformance_worker import execute
from audioenv.models.omni_driver import ModelConformanceRunner
from audioenv.models.scaffold import build_driver_scaffold
from audioenv.platform.api import create_app


MODEL_ID = "lab/sandboxed-omni"
IMAGE = "registry.example/audioenv-conformance@sha256:" + "a" * 64


def _report(model_id: str = MODEL_ID, *, passed: bool = True):
    checks = tuple(ConformanceCheckV1(name=name, passed=passed) for name in ModelConformanceRunner.check_names)
    return ModelConformanceReportV1(
        model_id=model_id, driver_id="lab/driver-v1", checks=checks,
        registered=passed, registration_state="conformant" if passed else "rejected",
    )


class _Transport:
    def __init__(self, report=None):
        self.report = report or _report()
        self.calls = []

    def run(self, **kwargs):
        self.calls.append(kwargs)
        return self.report


def test_scaffold_executes_but_cannot_pass_conformance(tmp_path):
    package = tmp_path / "driver.zip"
    package.write_bytes(build_driver_scaffold(MODEL_ID, "/models/new", "novel"))
    report = execute(package, MODEL_ID)
    assert report["registered"] is False
    assert report["registration_state"] == "rejected"
    assert len(report["checks"]) == len(ModelConformanceRunner.check_names)


def test_sandbox_rejects_wrong_model_and_incomplete_gate_set(tmp_path):
    package = tmp_path / "driver.zip"
    package.write_bytes(b"fixture")
    wrong = ModelConformanceSandbox(_Transport(_report("another/model")))
    try:
        wrong.validate(package_path=package, model_id=MODEL_ID, image=IMAGE)
    except RuntimeError as exc:
        assert "model_id" in str(exc)
    else:
        raise AssertionError("wrong model report was accepted")

    incomplete = ModelConformanceReportV1(
        model_id=MODEL_ID, driver_id="lab/driver-v1",
        checks=(ConformanceCheckV1(name="dependency_probe", passed=False),),
        registered=False, registration_state="rejected",
    )
    try:
        ModelConformanceSandbox(_Transport(incomplete)).validate(
            package_path=package, model_id=MODEL_ID, image=IMAGE,
        )
    except RuntimeError as exc:
        assert "exact required gates" in str(exc)
    else:
        raise AssertionError("incomplete report was accepted")


def test_api_promotes_only_complete_isolated_conformance(tmp_path):
    transport = _Transport()
    app = create_app(
        f"sqlite:///{tmp_path / 'api.sqlite'}", model_conformance_transport=transport,
        trusted_model_conformance_images=(IMAGE,),
    )
    client = TestClient(app)
    client.post("/api/models", json={
        "id": MODEL_ID, "display_name": "Sandboxed Omni",
        "adapter_type": "generated-needs-review", "role": "agent", "config": {},
    })
    package = build_driver_scaffold(MODEL_ID, "/models/new", "novel")
    uploaded = client.post(
        f"/api/models/{MODEL_ID}/driver-package", content=package,
        headers={"content-type": "application/zip"},
    )
    assert uploaded.status_code == 201
    result = client.post(
        f"/api/models/{MODEL_ID}/conformance-sandbox", json={"image": IMAGE},
    )
    assert result.status_code == 200, result.text
    body = result.json()
    assert body["model"]["available"] is True
    assert body["model"]["registration_state"] == "conformant"
    assert body["model"]["probe"]["runtime_image"] == IMAGE
    assert app.state.repository.get_artifact(body["artifact_id"]).kind == "model-conformance-report"
    assert transport.calls[0]["package_path"].is_file()


def test_api_leaves_model_unavailable_when_sandbox_cannot_run(tmp_path):
    class FailingTransport:
        def run(self, **kwargs):
            raise RuntimeError("rootless Docker is required")

    app = create_app(
        f"sqlite:///{tmp_path / 'failure.sqlite'}",
        model_conformance_transport=FailingTransport(),
        trusted_model_conformance_images=(IMAGE,),
    )
    client = TestClient(app)
    client.post("/api/models", json={
        "id": MODEL_ID, "display_name": "Sandboxed Omni",
        "adapter_type": "generated-needs-review", "role": "agent", "config": {},
    })
    client.post(
        f"/api/models/{MODEL_ID}/driver-package",
        content=build_driver_scaffold(MODEL_ID, "/models/new", "novel"),
        headers={"content-type": "application/zip"},
    )
    result = client.post(
        f"/api/models/{MODEL_ID}/conformance-sandbox", json={"image": IMAGE},
    )
    assert result.status_code == 422
    model = next(row for row in client.get("/api/models").json() if row["id"] == MODEL_ID)
    assert model["available"] is False
    assert model["registration_state"] == "generated_needs_review"


def test_api_rejects_user_controlled_conformance_image(tmp_path):
    transport = _Transport()
    app = create_app(
        f"sqlite:///{tmp_path / 'untrusted.sqlite'}",
        model_conformance_transport=transport,
        trusted_model_conformance_images=(IMAGE,),
    )
    client = TestClient(app)
    client.post("/api/models", json={
        "id": MODEL_ID, "display_name": "Sandboxed Omni",
        "adapter_type": "generated-needs-review", "role": "agent", "config": {},
    })
    client.post(
        f"/api/models/{MODEL_ID}/driver-package",
        content=build_driver_scaffold(MODEL_ID, "/models/new", "novel"),
        headers={"content-type": "application/zip"},
    )
    result = client.post(
        f"/api/models/{MODEL_ID}/conformance-sandbox",
        json={"image": "attacker.example/fake@sha256:" + "b" * 64},
    )
    assert result.status_code == 422
    assert "operator-approved" in result.text
    assert transport.calls == []


def test_api_rejects_tampered_driver_artifact(tmp_path):
    transport = _Transport()
    app = create_app(
        f"sqlite:///{tmp_path / 'tampered.sqlite'}",
        model_conformance_transport=transport,
        trusted_model_conformance_images=(IMAGE,),
    )
    client = TestClient(app)
    client.post("/api/models", json={
        "id": MODEL_ID, "display_name": "Sandboxed Omni",
        "adapter_type": "generated-needs-review", "role": "agent", "config": {},
    })
    uploaded = client.post(
        f"/api/models/{MODEL_ID}/driver-package",
        content=build_driver_scaffold(MODEL_ID, "/models/new", "novel"),
        headers={"content-type": "application/zip"},
    ).json()
    artifact = app.state.repository.get_artifact(uploaded["artifact_id"])
    Path(artifact.path).write_bytes(b"tampered")
    result = client.post(
        f"/api/models/{MODEL_ID}/conformance-sandbox", json={"image": IMAGE},
    )
    assert result.status_code == 422
    assert "content-hash" in result.text
    assert transport.calls == []


def test_docker_transport_requires_digest_and_applies_isolation(monkeypatch, tmp_path):
    package = tmp_path / "driver.zip"
    package.write_bytes(b"fixture")
    commands = []

    class Result:
        returncode = 0
        stderr = ""
        stdout = ""

    def fake_run(command, **kwargs):
        commands.append(command)
        result = Result()
        if command[1] == "info":
            result.stdout = '["name=rootless"]'
        else:
            result.stdout = json.dumps({"ok": True, "report": _report().model_dump(mode="json")})
        return result

    monkeypatch.setattr("audioenv.integrations.model_conformance_sandbox.shutil.which", lambda _: "/usr/bin/docker")
    monkeypatch.setattr("audioenv.integrations.model_conformance_sandbox.subprocess.run", fake_run)
    report = DockerModelConformanceTransport().run(
        package_path=package, model_id=MODEL_ID, image=IMAGE,
    )
    assert report.registered
    run = commands[-1]
    assert run[run.index("--network") + 1] == "none"
    assert "--read-only" in run and "no-new-privileges" in run and "--gpus" in run
    assert not any("docker.sock" in item for item in run)

    try:
        DockerModelConformanceTransport().run(
            package_path=package, model_id=MODEL_ID, image="mutable:latest",
        )
    except ValueError as exc:
        assert "immutable" in str(exc)
    else:
        raise AssertionError("mutable conformance image was accepted")


def test_dependency_image_build_copies_lock_but_not_driver_source(monkeypatch, tmp_path):
    package = tmp_path / "driver.zip"
    package.write_bytes(build_driver_scaffold(MODEL_ID, "/models/new", "novel"))
    dockerfiles = []

    class Result:
        returncode = 0
        stdout = '["name=rootless"]'
        stderr = ""

    def fake_run(command, **kwargs):
        if command[1] == "build":
            context = Path(command[-1])
            dockerfiles.append((context / "Dockerfile").read_text())
            Path(command[command.index("--iidfile") + 1]).write_text("sha256:" + "c" * 64)
        return Result()

    monkeypatch.setattr("audioenv.integrations.model_conformance_sandbox.shutil.which", lambda _: "/usr/bin/docker")
    monkeypatch.setattr("audioenv.integrations.model_conformance_sandbox.subprocess.run", fake_run)
    receipt = DockerModelConformanceImageBuilder().build(
        package_path=package, base_image=IMAGE,
    )
    assert receipt["image"] == "sha256:" + "c" * 64
    assert receipt["requirements_locked"] is False
    assert "driver.zip" not in dockerfiles[0]
    assert f"FROM {IMAGE}" in dockerfiles[0]


def test_api_builds_and_trusts_dependency_image(tmp_path):
    derived = "sha256:" + "d" * 64

    class Builder:
        def build(self, *, package_path, base_image):
            return {
                "schema_version": "audioenv.model-conformance-image/v1",
                "image": derived, "base_image": base_image,
                "driver_package_sha256": package_path.parent.name + package_path.name,
                "requirements_locked": False,
            }

    app = create_app(
        f"sqlite:///{tmp_path / 'builder.sqlite'}",
        model_conformance_image_builder=Builder(),
        trusted_model_conformance_images=(IMAGE,),
    )
    client = TestClient(app)
    client.post("/api/models", json={
        "id": MODEL_ID, "display_name": "Sandboxed Omni",
        "adapter_type": "generated-needs-review", "role": "agent", "config": {},
    })
    client.post(
        f"/api/models/{MODEL_ID}/driver-package",
        content=build_driver_scaffold(MODEL_ID, "/models/new", "novel"),
        headers={"content-type": "application/zip"},
    )
    result = client.post(
        f"/api/models/{MODEL_ID}/conformance-image", json={"base_image": IMAGE},
    )
    assert result.status_code == 201, result.text
    assert result.json()["image"] == derived
    assert derived in app.state.trusted_model_conformance_images
    assert app.state.repository.get_artifact(result.json()["artifact_id"]).kind == "model-conformance-image-receipt"
