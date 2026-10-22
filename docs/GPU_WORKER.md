# AudioEnv GPU worker contract

The provider-neutral worker entrypoint is:

```bash
audioenv-gpu-worker run /workspace/experiment.json
```

`experiment.json` must validate as `GPUWorkerExperimentV1`. The same bundle is
staged by local CUDA, RunPod, Modal, or a later customer-cluster provider.
Providers may allocate machines and transport files, but they must not rewrite
trainer configuration.

For local CUDA, AudioEnv persists a machine preflight before launch and then
runs the digest-qualified worker image with Docker as the invoking non-root UID, a read-only
root filesystem, no network, dropped capabilities, bounded PIDs, an ephemeral
`/tmp`, and only the selected AudioEnv data directory mounted at `/workspace`.
All model, dataset, and manifest paths must be contained by that mount (pinned
source checkouts may come from `/opt/audioenv/checkouts` in the image). Launch
still requires typing the exact run name. The control plane replays worker
events, imports only manifest-listed hash-matching artifacts, stops and removes
the container on terminal states, and records local cost as unavailable rather
than inventing a cloud price.

The worker writes `events.jsonl` and mirrors every line to stdout. Each line is
a `GPUWorkerEventV1` with a monotonically increasing sequence number. Event
types cover start, preflight, training, metrics, checkpoints, evaluation,
artifact manifest, cancellation, failure, and completion.
VeRL stdout/stderr is relayed as redacted `log` events; only explicit JSON
metric envelopes become `metric` events. The worker also persists an immutable
experiment snapshot and content-hashes a discovered checkpoint. Checkpoint
reload remains false until the separate evaluation gate verifies it.

For the Qwen path, VeRL runs the protected test split before optimization and
again at the final step. The worker then discovers the saved `global_step_`
checkpoint and launches a separate `val_only` process using
`resume_mode=resume_path`. It accepts the checkpoint only when the pre-reload
and reloaded sample outcomes match. Baseline and reloaded-trained results are
bound to identical environment, verifier, split-manifest, caller, audio,
sampling, and seed hashes before a paired comparison is exported.

AudioEnv patches the pinned VeRL FSDP optimizer inside each Ray worker to emit
per-rank gradient evidence. The run is rejected unless every configured actor
rank reports finite gradients exclusively on the four declared Thinker LoRA
targets and a changed trainable-parameter hash. Before advantage/loss
calculation, the controller compares rollout token log-probabilities with the
actor's recomputed values on response-mask tokens and rejects non-finite values
or drift above `logprob_max_abs_error` (default `0.05`). Both summaries are
content-hashed reproducibility artifacts.
Parameter hashes operate on dtype- and shape-bound raw tensor bytes, including
the release-configured `bfloat16`; they do not depend on NumPy representing the
training dtype or cast parameters to a lossy intermediary.

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
The agent loop intentionally returns no precomputed `reward_score`; otherwise
VeRL would create `rm_scores` immediately and skip the configured reward
manager. The protected result and expected verifier hash travel as tool-extra
fields, and VeRL's reward loop must invoke `compute_audioenv_score` before a
training reward exists.
The immutable-image contract command enables and verifies all four runtime
patches during the build: native Thinker/Talker output binding, trajectory
postprocessing, LoRA-gradient auditing, and rollout/training log-probability
parity. An upstream method move or inactive patch therefore fails the image
build before GPU provisioning.
Each rollout binds its dataset policy lineage to VeRL's required
`global_steps`, producing an exact `@verl-step-N` policy version plus equal
`min_global_steps` and `max_global_steps` fields. The pinned V1 trainer uses
those bounds for trajectory-span and off-policy-staleness accounting; missing
or invalid step metadata is rejected instead of silently assigning every
optimizer batch to the base-policy label.
The pinned native-audio bridge retains stage-0 Thinker tokens/log-probabilities
and the stage-2 Code2Wav waveform from the same vLLM-Omni request. It fails
closed if either output is absent, preventing an unrelated TTS waveform from
being attached to a trainable action. The bridge reads the pinned
`OmniRequestOutput.multimodal_output` contract directly: it accepts the
mapping-based `MultimodalPayload`, unwraps Code2Wav's per-request
`model_outputs`/`sr` values, and only then temporarily substitutes the retained
Thinker request for VeRL token processing.
Caller WAV paths remain immutable trace lineage, while the pinned Qwen audio
utility decodes and resamples them into waveform arrays. Those exact arrays are
used by both live rollout and VeRL's trainer-side multimodal reconstruction;
path strings are never passed to the Qwen feature extractor as if they were
waveforms.
The frozen caller service serializes inference and derives a stable RNG seed
from the episode seed and caller turn before every model invocation. Interleaved
episodes therefore retain independent histories and independent sampling
lineage instead of sharing process-global RNG order.

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

## RunPod transport lifecycle

The RunPod provider requires an immutable worker-image digest, a fresh live
quote, budget/runtime caps, and an exact typed run-name confirmation before it
can call the provisioning transport. After a Pod exists, the strict-host-key
OpenSSH transport performs the provider-neutral lifecycle:

1. upload the validated experiment plus small manifest/dataset inputs under
   `/workspace/runs/<experiment-id>`;
2. require model weights and large source checkouts to be pre-positioned under
   `/workspace` or the preserved `/runpod-volume` mount;
3. launch `audioenv-gpu-worker` under `nohup`, persist its PID/exit status, and
   replay contiguous typed events after control-plane reconnects;
4. request safe-boundary cancellation through the experiment sentinel;
5. download only safe manifest paths and verify every size and SHA-256 digest;
6. stop Pod compute on completion, failure, cancellation, or enforced limits,
   without deleting the separately managed network volume.

This lifecycle is covered by local transport tests. The REST and SSH paths
have not contacted RunPod, and no live Pod has been provisioned. Consequently
the provider remains integration-ready rather than GPU-accepted.

The FastAPI control plane persists the live quote and exact worker experiment
as one immutable review snapshot. Launch refreshes the quote and rejects price
or availability drift before provisioning. A ready Pod is not marked running
until that exact bundle has been staged and `audioenv-gpu-worker` has started.
The sync endpoint replays only unseen contiguous events into persisted logs and
metrics, imports hash-verified artifacts into content-addressed storage, drives
the experiment to its terminal state, and stops compute. The dashboard exposes
the same quote, exact-name confirmation, synchronization, and cancellation
workflow without treating mocked transport acceptance as real GPU acceptance.

Before model startup, the worker validates the disjoint trainer/caller CUDA
allocation against the attested `nvidia-smi` inventory. Every selected ordinal
must exist and report approximately 80 GB of memory; a four-GPU host therefore
cannot accidentally pass trainer preflight when the caller is assigned GPUs
4 and 5.

## Image build policy

[`docker/gpu-worker/Dockerfile`](../docker/gpu-worker/Dockerfile) has no default
base image. Release builds must supply an NVIDIA CUDA image using an immutable
digest, plus the built AudioEnv wheel:

```bash
docker/gpu-worker/build.sh \
  'nvidia/cuda@sha256:<approved-digest>' \
  'dist/audioenv_foundry-<version>-py3-none-any.whl' \
  'registry.example/audioenv-gpu:<release>' \
  'dist/gpu-worker-image-receipt.json'
```

The receipt path is optional and defaults to the path shown above. The build
wrapper rejects mutable base-image tags before Docker runs, verifies the wheel's
embedded version against the dependency lock, and passes every upstream commit
to Docker from that lock rather than relying on Dockerfile defaults. It labels
the image with the wheel and lock SHA-256 values, captures Docker's immutable
local image ID, and writes all of those identities and release pins to the
versioned JSON receipt. The wrapper is covered with a fake-Docker contract test.
GitHub Actions run `39747815437` also built the AMD64 image, executed the
pinned dependency and VeRL bridge contracts, and published it to GHCR.

The local Docker image ID is a content identity for that local image, but it is
not a pushed registry manifest digest. Publishing must separately record the
registry-returned digest, and AudioEnv providers continue to require that
digest-qualified image reference before a GPU launch.

The dependency receipt records the AudioEnv, Qwen checkpoint, VeRL-Omni, VeRL,
vLLM-Omni, tau2, and Repo2RLEnv revisions and is copied into the image at
`/opt/audioenv/dependencies.lock.json`.
After every package installation, the image also records the deterministically
sorted result of `pip freeze --all` at
`/opt/audioenv/python-environment.lock`. Runtime attestation refuses to start
Qwen when this resolved lock is absent, empty, or unsorted, includes its hash
and distribution count in the preflight evidence, and copies the exact file
into the run artifacts after verifying that its hash did not change. This
captures transitive packages selected during the image build without claiming
that a later rebuild from the source-pin receipt alone is byte-identical.
The build also retains detached, commit-verified VeRL-Omni and tau2 source
checkouts under `/opt/audioenv/checkouts`; runtime attestation and the launcher
use those exact paths. Tau2 is installed in the worker rather than merely
named in the receipt. The image defaults to `sleep infinity` so an SSH-managed
Pod remains alive; every provider supplies the complete worker command and
cannot accidentally combine it with a duplicate image entrypoint.
The published build is
`ghcr.io/ssdeb-rr/audioenv-gpu@sha256:7c25d71104883f8194b0626df286462d3a1a536c12e1dfc5f8eaf6e11da6d26e`.
An anonymous registry manifest request returned HTTP 200, and its config reports
`linux/amd64`, user `audioenv`, and `sleep infinity`. This proves image
readiness, not GPU acceptance: no NVIDIA device, Qwen checkpoint, or optimizer
step was exercised during the GitHub-hosted build.

The manual `GPU worker image` GitHub Actions workflow provides the release
build boundary on an AMD64 runner. It fixes the approved CUDA manifest digest
in version control, invokes the same `build.sh` contract (including
`audioenv-verl-contract-check`), publishes to GHCR, and uploads a receipt bound
to the registry digest. Running this workflow builds and publishes an image;
it never calls RunPod or provisions paid compute. The package must be made
public before RunPod can pull it without registry credentials.

The pinned VeRL-Omni revision deliberately overrides one stale dependency
declaration in its pinned vLLM-Omni revision: VeRL-Omni requires
`accelerate>=1.14.0`, while vLLM-Omni metadata still says
`accelerate==1.12.0`. `check_dependencies.py` accepts only that exact,
version-bound upstream conflict and fails the image for every additional
`pip check` result. The runtime contract check remains a separate build gate.
