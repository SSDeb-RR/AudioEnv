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
- The Harness Foundry accepts a local path or HTTPS GitHub repository, safely
  inventories it, compiles validated `audioenv-task.json` inputs, recognizes
  AudioEnv benchmark manifests, and emits a protected OCI scaffold for coding
  repositories without importing or executing repository code.
- Users can explicitly select Codex-assisted authoring. AudioEnv passes Codex
  a bounded redacted digest, requires schema-constrained declarative output,
  runs the candidate through deterministic validation, and keeps it in
  `generated_needs_review`. No Codex call occurs implicitly.
- Model onboarding detects the Qwen3-Omni family, selects its shared driver,
  or creates a reviewable driver-conformance scaffold for an unknown model.
  Caller and trainer choices in the run wizard are resolved from declared
  capabilities instead of a hard-coded menu.
- The Foundry can invoke Repo2RLEnv `v0.9.3` through a restricted Python 3.12
  `uv` worker for the `pr_diff` pipeline and import its Harbor task output.
  Docker-dependent pipelines are rejected rather than mounting the host socket.
  Harbor instructions remain public
  while verifier paths and hashes are stored in a protected artifact.
- Generated environments use an explicit promotion report. Sandbox build,
  oracle, no-op, adversarial, determinism, state-isolation, leakage,
  reward-integrity, and verifier-immutability gates must all pass.
- `TrajectoryV2` validates multimodal feature references, prompt/response
  positions, response-only masks, rollout log-probabilities, policy version,
  audio lineage, episode termination, and reward provenance.
- Distinct Qwen3-Omni Thinker trainer plugins now emit GSPO and GRPO loss
  configurations against pinned VeRL-Omni v0.2. Both use grouped GRPO outcome
  advantages and reject groups smaller than two.

## Partial

- τ² retail is pinned to `fc0055dc4e0a316c3f83133267fbd6faaa770992`.
  A Python 3.12 JSON-lines worker implements discovery, reset, tool execution,
  and environment-state verification, but Python 3.12 is unavailable on this
  host. Full upstream communication-reward parity is not claimed.
- `AudioEnvAgentLoop` validates policy version, masks, token/log-probability
  alignment, grouped rollouts, zero-variance rate, and VeRL-shaped V2 batch
  export. It has not run inside the pinned CUDA VeRL worker.
- RunPod has REST v2 quote/provision/state/stop methods, exact-name confirmation,
  runtime/budget stops, reconnect, and SFTP command construction. Live API,
  SSH event streaming, and artifact collection are not verified.

## Blocked

- Qwen3-Omni-30B-A3B native caller/agent execution and VeRL-Omni GSPO/GRPO LoRA
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
60 passed

uv run audioenv validate
341 passed, 0 failed

cd web && npm run build && npm run test:e2e
Vite production build passed
2 Playwright product-flow tests passed

uv build
audioenv_foundry-0.5.0 wheel and sdist built
```

The product-flow gates cover project creation, resource selection, preflight,
explicit launch, worker completion, real parameter updates, checkpoint reload,
two held-out evaluations, compatible paired comparison, artifact download
exposure, and SPA deep-link serving.
