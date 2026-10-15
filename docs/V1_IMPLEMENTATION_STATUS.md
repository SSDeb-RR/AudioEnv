# AudioEnv V1 implementation status

Verified on 2026-10-16 on an Apple M4 MacBook Air with 16 GB RAM and no CUDA
device. `working` means executed successfully; `partial` means a real boundary
exists but its external gate has not passed; `blocked` requires unavailable
hardware, credentials, or approval; `missing` means no implementation exists.

## Working

- A provider-neutral `audioenv-gpu-worker run experiment.json` command now
  consumes an immutable typed bundle, emits versioned JSONL lifecycle events,
  supports safe-boundary cancellation, and writes a content-hashed artifact
  manifest. Its local backend has completed a real optimizer/checkpoint/reload/
  evaluation run; the Qwen backend remains CUDA-gated.

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
- A Modal Sandbox provider uses the same immutable GPU-worker experiment and
  event contracts. It implements quote hashing, exact-name confirmation,
  gVisor runtime selection, named-secret references, volume staging,
  contiguous event replay, budget/runtime stops, cancellation, reconnect, and
  hash-verified artifact recovery behind an injected transport. Its live SDK
  transport and paid acceptance remain intentionally blocked.
- The Harness Foundry accepts a local path or HTTPS GitHub repository, safely
  inventories it, compiles validated `audioenv-task.json` inputs, recognizes
  AudioEnv benchmark manifests, and emits a protected OCI scaffold for coding
  repositories without importing or executing repository code.
- Users can explicitly select Codex-assisted authoring. AudioEnv passes Codex
  a bounded redacted digest, requires schema-constrained declarative output,
  runs the candidate through deterministic validation, and keeps it in
  `generated_needs_review`. No Codex call occurs implicitly.
- Model onboarding detects the Qwen3-Omni family, selects its shared driver,
  or creates a deterministic, downloadable Python driver SDK for an unknown
  model. The generated package contains a manifest, driver skeleton, and test,
  but truthfully reports `generated_needs_review` and remains unavailable.
  Caller and trainer choices in the run wizard are resolved from declared
  capabilities instead of a hard-coded menu.
- The executable Omni conformance contract now checks native-audio inference,
  structured actions, token/log-probability alignment, multimodal position
  channels, response masks, declared gradient scope, a real optimizer-state
  and parameter-hash change, checkpoint round-trip, and deterministic probing.
  A conformant fixture passes every gate; Qwen remains unregistered until the
  same suite runs in the pinned CUDA worker.
- The Foundry can invoke Repo2RLEnv `v0.9.3` through a restricted Python 3.12
  `uv` worker for the `pr_diff` pipeline and import its Harbor task output.
  Docker-dependent pipelines are rejected rather than mounting the host socket.
  Harbor instructions remain public
  while verifier paths and hashes are stored in a protected artifact.
- Environment promotion no longer accepts caller-supplied gate booleans. The
  backend aggregates persisted validator checks, hashes the protected verifier
  artifact, issues content-addressed HMAC-signed evidence with a locally held
  `0600` key, and revalidates its signature and environment hash immediately
  before registry promotion. Sandbox/build, oracle, no-op, adversarial,
  determinism, state-isolation, leakage, reward-integrity, and
  verifier-immutability gates must all pass.
- `TrajectoryV2` validates multimodal feature references, prompt/response
  positions, response-only masks, rollout log-probabilities, policy version,
  audio lineage, episode termination, and reward provenance. Training turns
  preserve either flat positions or Qwen-style multi-channel multimodal RoPE
  positions end to end instead of flattening them at the trainer boundary.
- Distinct Qwen3-Omni Thinker trainer plugins now emit the pinned VeRL-Omni
  V1 `main_omni` recipe for GSPO and GRPO. The command selects the actual
  vLLM-Omni Qwen pipeline, FSDP2, Thinker-only LoRA targets, grouped GRPO
  advantages, and a protected AudioEnv reward function. The prior obsolete
  config-path launcher and fictitious hook environment variables were removed.

## Partial

- τ² retail is pinned to `fc0055dc4e0a316c3f83133267fbd6faaa770992`.
  A process-owning client now runs its Python 3.12 JSON-lines worker through
  `uv`, correlates requests, enforces deadlines, captures bounded diagnostics,
  and terminates the process group safely. A live pinned run discovered 114
  retail tasks, reset task `0`, exposed 16 tools, and produced a `0.0` no-op
  environment reward. The worker now keeps the private user scenario behind a
  separate caller-only operation, records agent/user semantic turns and tool
  results, and invokes τ²'s official `EvaluationType.ALL` evaluator after an
  explicit terminal action. A live no-op episode returned the complete
  upstream reward breakdown at `0.0`. Official train/test membership is
  preserved instead of being overwritten by the aggregate `base` list. Gold
  trajectory parity across every reward basis is not yet claimed.
- Declarative in-process environments now receive backend-issued promotion
  evidence only after a constrained subprocess independently reconstructs the
  environment and reproduces both its environment and verifier hashes. The
  worker uses a sanitized environment, bounded input and deadline, closed
  descriptors, and host-supported POSIX resource limits. Harbor/repository
  candidates still require an isolated OCI validation worker before they can
  produce equivalent evidence.
- `AudioEnvAgentLoop` validates policy version, masks, token/log-probability
  alignment, grouped rollouts, zero-variance rate, and VeRL-shaped V2 batch
  export. `AudioEnvVerlAgentLoop` now subclasses and registers against the
  pinned VeRL `AgentLoopBase` when loaded in the CUDA worker, converts an
  append-only `TrajectoryV2` into VeRL's exact prompt/response/mask/log-prob
  output, and masks caller/tool observation tokens to zero. It refuses an
  implicit scripted fallback. The reviewed Qwen caller/tau2 runtime factory
  that drives VeRL's live policy server is still not implemented, and this
  loop has not run inside the GPU worker.
- RunPod has REST v2 quote/provision/state/stop methods, exact-name confirmation,
  runtime/budget stops, reconnect, and SFTP command construction. Live API,
  SSH event streaming, and artifact collection are not verified.
- Modal has a fully tested control-plane contract but not a concrete live SDK
  transport or external Sandbox acceptance run.

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
  billing, Kubernetes, and additional cloud providers beyond RunPod and Modal
  are outside V1.

## Verification evidence

```text
uv run --extra dev --extra platform --extra train pytest -q
85 passed

uv run audioenv validate
389 passed, 0 failed

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
