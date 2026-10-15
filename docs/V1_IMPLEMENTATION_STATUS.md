# AudioEnv V1 implementation status

Verified on 2026-10-15 on an Apple M4 MacBook Air with 16 GB RAM and no CUDA
device. `working` means executed successfully; `partial` means a real boundary
exists but its external gate has not passed; `blocked` requires unavailable
hardware, credentials, or approval; `missing` means no implementation exists.

## Working

- `audioenv app --data-dir .audioenv` applies Alembic migrations, recovers
  abandoned jobs, and serves FastAPI plus the compiled React SPA from one
  loopback URL. The wheel includes the SPA and migrations, so Node is required
  only to rebuild frontend assets.
- Projects, registries, configuration hashes, preflight snapshots, lifecycle
  events, metrics, worker logs, evaluations, samples, comparisons, and
  artifacts persist in SQLite.
- The dashboard has Projects, Models, Benchmarks & Tasks, Environments,
  Training Runs, Evaluations, Artifacts, and Settings. Its run wizard selects
  persisted resources, performs preflight, shows hardware/cost, and requires
  explicit launch confirmation.
- Reference V2 has 48 tasks: 32 train, 8 validation, and 8 test, evenly split
  between retail and telecom. Membership and seeds are content-hashed.
- `SmallAudioToolPolicyTrainer` performs real PyTorch REINFORCE updates from
  WAV features and public action candidates. It never reads oracle actions,
  hidden facts, expected state, or verifier predicates.
- Baseline evaluation runs before optimization. Trained evaluation runs only
  after checkpoint reload. Both use the same 24 held-out samples: 8 test tasks
  under clean, degraded, and fast-speaking audio.
- Comparison rejects drift in environment, verifier, split, caller, audio
  pipeline, sampling, or seeds and reports a paired bootstrap interval.
- Checkpoint and manifest downloads resolve by content-addressed artifact ID;
  arbitrary paths are not accepted.
- RunPod quote, typed-confirmation, provisioning, cancellation, and budget-stop
  contracts are covered with a fake transport. Tests and startup never create
  paid compute.

## Partial

- τ² retail is pinned to `fc0055dc4e0a316c3f83133267fbd6faaa770992`.
  A Python 3.12 JSON-lines worker implements discovery, reset, tool execution,
  and environment-state verification, but Python 3.12 is unavailable on this
  host. Full upstream communication-reward parity is not claimed.
- `AudioEnvAgentLoop` validates policy version, masks, token/log-probability
  alignment, isolated rollouts, and VeRL-shaped batch export. It has not run
  inside the pinned CUDA VeRL worker.
- RunPod has a guarded provider contract and truthful UI state. Live REST v2,
  SSH streaming, SFTP collection, and reconnect behavior are not verified.

## Blocked

- Qwen3-Omni-30B-A3B native caller/agent execution and VeRL-Omni GSPO LoRA
  require approved Linux CUDA infrastructure, model weights, credentials, a
  live quote, maximum budget/runtime, and typed launch confirmation.
- The UI reports `Integration ready; GPU acceptance not run`.
- Native-audio learning lift, independent Qwen caller transfer, GPU utilization,
  and paid cost remain blocked until that run succeeds.

## Missing / out of scope

- Repository execution remains unavailable until an OCI sandbox passes the
  specified network, privilege, filesystem, resource, socket, and secret tests.
- Full duplex, interruption, telephony, Talker gradients, multi-tenancy,
  billing, Kubernetes, and additional cloud providers are outside V1.

## Verification evidence

```text
uv run --extra dev --extra platform --extra train pytest -q
46 passed

uv run audioenv validate
341 passed, 0 failed

cd web && npm run build && npm run test:e2e
Vite production build passed
1 Playwright product-flow test passed

uv build
audioenv_foundry-0.3.0 wheel and sdist built
```

The product-flow gates cover project creation, resource selection, preflight,
explicit launch, worker completion, real parameter updates, checkpoint reload,
two held-out evaluations, compatible paired comparison, artifact download
exposure, and SPA deep-link serving.
