# AudioEnv GPU worker contract

The provider-neutral worker entrypoint is:

```bash
audioenv-gpu-worker run /workspace/experiment.json
```

`experiment.json` must validate as `GPUWorkerExperimentV1`. The same bundle is
staged by local CUDA, RunPod, Modal, or a later customer-cluster provider.
Providers may allocate machines and transport files, but they must not rewrite
trainer configuration.

The worker writes `events.jsonl` and mirrors every line to stdout. Each line is
a `GPUWorkerEventV1` with a monotonically increasing sequence number. Event
types cover start, preflight, training, metrics, checkpoints, evaluation,
artifact manifest, cancellation, failure, and completion.

The local CPU proof backend is executable on development machines. The
`qwen3-omni-verl` backend validates the pinned VeRL checkout, checkpoint,
caller, rollout seed data, environment manifest, CUDA topology, and resume
checkpoint before launching VeRL-Omni. It exits rather than silently ignoring
an unsupported configuration.

Cancellation is requested by creating the configured cancellation sentinel.
The CPU trainer checks it at optimizer-step boundaries; the VeRL subprocess is
terminated, then killed after a bounded grace period if necessary.

The worker hashes every output file into `worker-artifacts.json`. Cloud
providers collect this manifest and its referenced files; the control plane
registers them by content hash.

## Image build policy

[`docker/gpu-worker/Dockerfile`](../docker/gpu-worker/Dockerfile) has no default
base image. Release builds must supply an NVIDIA CUDA image using an immutable
digest, plus the built AudioEnv wheel:

```bash
docker build \
  --build-arg BASE_IMAGE='nvidia/cuda@sha256:<approved-digest>' \
  --build-arg AUDIOENV_WHEEL='dist/audioenv_foundry-<version>-py3-none-any.whl' \
  -f docker/gpu-worker/Dockerfile .
```

The dependency receipt records the VeRL-Omni, tau2, and Repo2RLEnv revisions.
Building and publishing the release image remains blocked until a CUDA base
digest is selected and the image passes the GPU acceptance suite.
