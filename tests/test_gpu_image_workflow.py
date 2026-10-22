from pathlib import Path


WORKFLOW = Path(".github/workflows/gpu-worker-image.yml")
CUDA_DIGEST = "sha256:36c66a3ad4608580cf937f7ec3add9323a610956cfe8c9e2f99ef2ea0c896f01"


def test_gpu_image_workflow_is_manual_pinned_and_non_provisioning():
    source = WORKFLOW.read_text()

    assert "workflow_dispatch:" in source
    assert "schedule:" not in source
    assert "push:" not in source
    assert "packages: write" in source
    assert CUDA_DIGEST in source
    assert "test \"$(uname -m)\" = \"x86_64\"" in source
    assert "docker/gpu-worker/build.sh" in source
    assert "audioenv-verl-contract-check" not in source  # build.sh/Dockerfile own this gate
    assert "docker push" in source
    assert "registry_digest" in source
    assert "api.runpod" not in source.casefold()
    assert "create_pod" not in source
