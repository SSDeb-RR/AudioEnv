import io
import json
import zipfile

import pytest
from fastapi.testclient import TestClient

from audioenv.models.package import inspect_driver_package
from audioenv.models.scaffold import build_driver_scaffold
from audioenv.platform.api import create_app


MODEL_ID = "lab/novel-omni"


def _rewrite(payload: bytes, updates: dict[str, object] | None = None, extra=None) -> bytes:
    source = zipfile.ZipFile(io.BytesIO(payload))
    files = {name: source.read(name) for name in source.namelist()}
    if updates:
        manifest = json.loads(files["manifest.json"])
        manifest.update(updates)
        files["manifest.json"] = json.dumps(manifest).encode()
    if extra:
        files[extra[0]] = extra[1]
    target = io.BytesIO()
    with zipfile.ZipFile(target, "w") as archive:
        for name, value in files.items():
            archive.writestr(name, value)
    return target.getvalue()


def test_driver_package_intake_binds_model_module_and_all_gates():
    payload = build_driver_scaffold(MODEL_ID, "/models/novel", "novel")
    report = inspect_driver_package(payload, expected_model_id=MODEL_ID)
    assert report.sha256
    assert report.manifest["registration_state"] == "generated_needs_review"
    assert any(name.endswith("/driver.py") for name in report.files)

    weakened = _rewrite(payload, {"required_conformance": ["dependency_probe"]})
    with pytest.raises(ValueError, match="weaken"):
        inspect_driver_package(weakened, expected_model_id=MODEL_ID)
    with pytest.raises(ValueError, match="model_id"):
        inspect_driver_package(payload, expected_model_id="another/model")


def test_driver_package_rejects_traversal_native_code_and_unlocked_dependencies():
    payload = build_driver_scaffold(MODEL_ID, "/models/novel", "novel")
    with pytest.raises(ValueError, match="unsafe"):
        inspect_driver_package(
            _rewrite(payload, extra=("../escape.py", b"bad")), expected_model_id=MODEL_ID,
        )
    with pytest.raises(ValueError, match="compiled"):
        inspect_driver_package(
            _rewrite(payload, extra=("plugin.so", b"binary")), expected_model_id=MODEL_ID,
        )
    with pytest.raises(ValueError, match="hash-pinned"):
        inspect_driver_package(
            _rewrite(payload, extra=("requirements.lock", b"transformers>=5\n")),
            expected_model_id=MODEL_ID,
        )


def test_driver_package_api_persists_artifact_but_does_not_register(tmp_path):
    app = create_app(f"sqlite:///{tmp_path / 'api.sqlite'}")
    client = TestClient(app)
    registered = client.post("/api/models", json={
        "id": MODEL_ID, "display_name": "Novel Omni",
        "adapter_type": "generated-needs-review", "role": "agent", "config": {},
    })
    assert registered.status_code == 201
    payload = build_driver_scaffold(MODEL_ID, "/models/novel", "novel")
    response = client.post(
        f"/api/models/{MODEL_ID}/driver-package", content=payload,
        headers={"content-type": "application/zip"},
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["model"]["available"] is False
    assert body["model"]["registration_state"] == "generated_needs_review"
    assert body["model"]["config"]["driver_package"]["artifact_id"] == body["artifact_id"]
    artifact = app.state.repository.get_artifact(body["artifact_id"])
    assert artifact.kind == "model-driver-package"


def test_certified_driver_cannot_be_replaced(tmp_path):
    client = TestClient(create_app(f"sqlite:///{tmp_path / 'certified.sqlite'}"))
    payload = build_driver_scaffold(
        "audioenv/small-audio-tool-policy", "/models/tiny", "tiny",
    )
    response = client.post(
        "/api/models/audioenv/small-audio-tool-policy/driver-package",
        content=payload, headers={"content-type": "application/zip"},
    )
    assert response.status_code == 409


def test_driver_upload_rejects_oversized_body_before_zip_processing(tmp_path):
    client = TestClient(create_app(f"sqlite:///{tmp_path / 'limit.sqlite'}"))
    client.post("/api/models", json={
        "id": MODEL_ID, "display_name": "Novel Omni",
        "adapter_type": "generated-needs-review", "role": "agent", "config": {},
    })
    response = client.post(
        f"/api/models/{MODEL_ID}/driver-package",
        content=b"x" * (4 * 1024 * 1024 + 1),
        headers={"content-type": "application/zip"},
    )
    assert response.status_code == 413
