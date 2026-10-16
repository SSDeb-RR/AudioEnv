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
VeRL stdout/stderr is relayed as redacted `log` events; only explicit JSON
metric envelopes become `metric` events. The worker also persists an immutable
experiment snapshot and content-hashes a discovered checkpoint. Checkpoint
reload remains false until the separate evaluation gate verifies it.

The local CPU proof backend is executable on development machines. The
`qwen3-omni-verl` backend performs two fail-closed preflights. Runtime
attestation compares the experiment pins with the dependency receipt embedded
in the image, reads the installed Git revisions for VeRL-Omni, VeRL, and
vLLM-Omni, verifies the exact Qwen checkpoint revision, checks the pinned tau2
checkout, hashes the train/test JSONL and manifests, rejects task overlap, and
probes PyTorch CUDA. Trainer preflight then checks the VeRL checkout, CUDA
topology, model/caller paths, environment manifest, and resume checkpoint. The
worker persists the complete evidence and exits before starting model services
when either preflight fails.

The Qwen launcher mirrors the pinned VeRL-Omni V1 recipe: it invokes
`verl_omni.trainer.main_omni`, selects the `qwen3_omni_moe` vLLM-Omni
full pipeline for rollouts, uses FSDP2, excludes the Talker, codec, visual, and
audio-tower modules from optimization, and applies LoRA only to `q_proj`,
`k_proj`, `v_proj`, and `o_proj`.
VeRL loads `audioenv.training.verl_plugin` in its driver and Ray workers. The
plugin exposes a hash-bound reward function that ignores model-printed answers
and accepts only the protected verifier result produced by the environment.
The pinned native-audio bridge retains stage-0 Thinker tokens/log-probabilities
and the stage-2 Code2Wav waveform from the same vLLM-Omni request. It fails
closed if either output is absent, preventing an unrelated TTS waveform from
being attached to a trainable action.

This correct launch and reward boundary does not by itself prove the complete
multi-turn Qwen caller/agent loop. That remains a GPU acceptance item until the
agent loop has run against the pinned VeRL commit and emitted a validated
native-audio trajectory.

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
docker/gpu-worker/build.sh \
  'nvidia/cuda@sha256:<approved-digest>' \
  'dist/audioenv_foundry-<version>-py3-none-any.whl' \
  'registry.example/audioenv-gpu:<release>'
```

The build wrapper rejects mutable base-image tags before Docker runs. The
dependency receipt records the AudioEnv, Qwen checkpoint, VeRL-Omni, VeRL,
vLLM-Omni, tau2, and Repo2RLEnv revisions and is copied into the image at
`/opt/audioenv/dependencies.lock.json`.
Building and publishing the release image remains blocked until a CUDA base
digest is selected and the image passes the GPU acceptance suite.
