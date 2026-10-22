from pathlib import Path


WORKFLOW = Path(".github/workflows/gpu-worker-image.yml")
DOCKERFILE = Path("docker/gpu-worker/Dockerfile")
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


def test_gpu_worker_installs_git_before_fetching_pinned_sources():
    source = DOCKERFILE.read_text()

    install = source.index("apt-get install")
    git_probe = source.index("git --version")
    first_checkout = source.index("git -C /opt/audioenv/checkouts/verl-omni init")
    assert install < git_probe < first_checkout
    assert "ca-certificates" in source[install:first_checkout]


def test_gpu_worker_bootstraps_pinned_python_312_before_dependency_install():
    source = DOCKERFILE.read_text()

    assert "ARG UV_VERSION=0.9.7" in source
    assert "ARG UV_SHA256=b26fcc8dfa1c39b5a5613445af3be3eefda45d9a39359bee271eafe34913583e" in source
    assert "ARG PYTHON_VERSION=3.12.11" in source
    assert "sha256sum --check --strict" in source
    python_probe = source.index("sys.version_info[:2] == (3, 12)")
    dependency_install = source.index('"vllm-omni @ git+')
    assert python_probe < dependency_install
    assert 'ENV PATH="/opt/audioenv/venv/bin:${PATH}"' in source


def test_gpu_worker_preserves_wheel_name_and_rejects_dependency_conflicts():
    source = DOCKERFILE.read_text()

    assert "ARG ACCELERATE_VERSION=1.12.0" in source
    assert "COPY ${AUDIOENV_WHEEL} /tmp/audioenv-wheels/" in source
    assert "/tmp/audioenv-wheels/*.whl" in source
    assert "/tmp/audioenv.whl" not in source
    constraints = source.index("pip-constraints.txt")
    verl_omni = source.index('"verl-omni[audio] @ git+')
    pip_check = source.index("python -m pip check")
    environment_lock = source.index("python -m pip freeze --all")
    assert constraints < verl_omni < pip_check < environment_lock
