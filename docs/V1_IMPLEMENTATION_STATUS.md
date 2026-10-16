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
- The Qwen worker now fails closed on immutable runtime attestation before
  starting the caller or trainer. It compares the experiment with the image's
  embedded dependency receipt, verifies installed Git revisions, the exact
  Qwen checkpoint revision, pinned tau2 checkout, CUDA runtime, and separately
  hashed/disjoint train and held-out manifests. The image build wrapper rejects
  mutable base tags. Building that image still awaits an approved CUDA digest.
- The worker now snapshots the immutable experiment, relays real JSON metric
  envelopes and redacted stdout/stderr as versioned events while preserving
  local logs, detects and content-hashes the resulting VeRL checkpoint, and
  marks reload verification honestly as pending until evaluation runs.
- Local CUDA now uses that same bundle through a guarded Docker lifecycle:
  immutable image digest, non-root/read-only/network-off execution, mounted
  input containment, runtime timeout, typed event replay, reconnect,
  cancellation, container cleanup, and hash-verified artifact import. Its
  transport and persisted API lifecycle are executable-test verified; this
  Mac cannot run the CUDA acceptance path.

- `audioenv app --data-dir .audioenv` applies Alembic migrations, recovers
  abandoned jobs, and serves FastAPI plus the compiled React SPA from one
  loopback URL. The wheel includes the SPA and migrations, so Node is required
  only to rebuild frontend assets.
- Projects, registries, configuration hashes, preflight snapshots, lifecycle
  events, metrics, worker logs, evaluations, samples, comparisons, and
  artifacts persist in SQLite.
- Generic model and experiment APIs reject embedded passwords, API keys,
  tokens, credentials, private keys, and secret values; callers must provide
  environment/provider secret references instead. Public experiment, model,
  compute-resource, log, artifact, overview, SSE, and evaluation responses also
  redact unsafe legacy fields so previously persisted values cannot leak into
  the dashboard or API.
- The dashboard has Projects, Models, Benchmarks & Tasks, Environments,
  Training Runs, Evaluations, Artifacts, and Settings. Its run wizard selects
  persisted resources, performs preflight, shows hardware/cost, and requires
  explicit launch confirmation.
- Reference V2 has 48 tasks: 32 train, 8 validation, and 8 test, evenly split
  between retail and telecom. Membership and seeds are content-hashed.
- `SmallAudioToolPolicyTrainer` performs real PyTorch REINFORCE updates from
  WAV features and public action candidates. It never reads oracle actions,
  hidden facts, expected state, or verifier predicates.
- Every local training episode is streamed as a public, observability-redacted
  JSONL trace and registered as a content-hashed experiment artifact. The
  run-scoped API verifies that hash before returning bounded records, and the
  dashboard displays actual actions, reward events, state hashes, latency, and
  terminal outcome. Tampered trace artifacts fail closed.
- During GPU trainer and checkpoint-reload processes, the worker samples real
  per-device utilization, used/total memory, and power through `nvidia-smi`
  every five seconds. Missing NVIDIA tooling or malformed device rows produce
  no metric rather than a fabricated zero. The run page shows persisted GPU
  telemetry with its metric name and source, and labels unreported utilization,
  memory, throughput, and measured cost as unavailable.
- Run details expose the persisted preflight, gradient-scope, parameter-change,
  log-probability, checkpoint-reload, and evaluation event payloads as training
  integrity evidence. The model registry likewise exposes the stored probe and
  conformance report rather than reducing it to an availability badge.
- The VeRL postprocessor now persists each validated native `TrajectoryV2` to
  a per-process JSONL file only after Qwen processor positions and audio spans
  are bound. Local CUDA, RunPod, and Modal artifact import classify these files
  through the same `training_trajectories` contract. The trace API normalizes a
  safe public view without returning token IDs, log-probabilities, private
  caller state, or observability transcripts; unavailable native latency is
  explicitly marked in trace metrics. Real GPU production remains unrun.
- Baseline evaluation runs before optimization. Trained evaluation runs only
  after checkpoint reload. Both use the same 24 held-out samples: 8 test tasks
  under clean, degraded, and fast-speaking audio.
- Comparison rejects drift in environment, verifier, split, caller, audio
  pipeline, sampling, or seeds and reports a paired bootstrap interval.
- Every evaluation row opens a detail view showing its checkpoint, environment,
  verifier, and protocol hashes, exact evaluation configuration, and persisted
  per-task/audio-profile/seed outcomes. Aggregate claims are therefore
  traceable to all 24 held-out samples rather than only a headline percentage.
- Checkpoint and manifest downloads resolve by content-addressed artifact ID;
  arbitrary paths are not accepted. All artifact kinds are rehashed before
  download, not only checkpoints and trajectories, so corrupted evaluation,
  dependency, evidence, or export files fail closed.
- The experiment export contains its immutable configuration hash, a stable
  hash over the complete artifact manifest, SHA-256 artifact identities,
  checkpoint lineage, and an explicit `complete`/`incomplete` status with
  missing reproduction inputs. Large checkpoints remain individually
  content-addressed instead of being synchronously duplicated into a ZIP.
- RunPod quote, typed-confirmation, provisioning, cancellation, budget-stop,
  strict-host-key SSH launch, reconnectable event replay, cancellation
  sentinel, immutable input staging, and hash-verified artifact recovery are
  covered with fake transports. Large model/checkpoint and source directories
  must be pre-positioned on the Pod workspace or persistent volume. Tests and
  startup never create paid compute.
- The persisted product workflow now validates and binds a complete
  `GPUWorkerExperimentV1` to each quote, refreshes the quote before paid
  provisioning, stages and launches the exact bundle only after the Pod is
  ready, replays typed events into database logs/metrics, imports verified
  artifacts, supports reconnect and cancellation, and stops compute on every
  terminal worker event. The dashboard exposes configuration, quote review,
  exact typed approval, resource status, sync, and stop controls.
- A Modal Sandbox provider uses the same immutable GPU-worker experiment and
  event contracts. It implements quote hashing, exact-name confirmation,
  gVisor runtime selection, named-secret references, volume staging,
  contiguous event replay, budget/runtime stops, cancellation, reconnect, and
  hash-verified artifact recovery behind an injected transport. A lazy official
  Python SDK transport now maps named Apps, registry images, Volumes, Secrets,
  GPU gVisor Sandboxes, stdout events, reconnect, termination, and file
  recovery without importing Modal or making calls at startup. The installed
  `modal==1.6.2` signatures were checked locally and the mapping is fake-SDK
  tested. The FastAPI lifecycle and dashboard now expose persisted preflight,
  an explicitly operator-configured price estimate, exact-name confirmation,
  launch, reconnect/sync, metrics, artifact recovery, and cancellation. Modal
  credentials and paid Sandbox acceptance were not used.
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
- Model registration now persists a separate `certified`, `conformant`,
  `generated_needs_review`, or `rejected` state. A successful dependency probe
  alone cannot mark a model usable; only the built-in certified local proof or
  a complete conformance report can set `available=true`. Migration `0004`
  backfills legacy databases, and the dashboard shows the actual registration
  state rather than collapsing it into a boolean.
- The Models product surface now accepts an implemented driver ZIP for a
  review-gated model. Backend intake enforces a 4 MiB compressed/16 MiB
  expanded ceiling, safe relative paths, no symlinks or native binaries,
  exact model/module binding, the complete ordered conformance gate list, and
  hash-pinned optional dependencies. The content-addressed package remains
  `generated_needs_review`; upload never executes code or marks the model
  available.
- Uploaded drivers now have an executable rootless-OCI conformance action in
  the API and Models screen. It requires an immutable image digest from the
  operator-controlled `AUDIOENV_MODEL_CONFORMANCE_IMAGES` allowlist, runs with
  network disabled, a read-only filesystem, dropped capabilities, no host
  socket, bounded CPU/memory/PIDs, GPU access, and an optional read-only model
  mount. The worker imports the driver only inside that sandbox, runs the exact
  ten-gate suite, persists the content-addressed report, and changes the model
  to `conformant` only if every gate passes. Fake-container/API tests and a real
  rejected-scaffold worker execution pass locally; a real custom Omni image and
  GPU checkpoint remain externally gated.
- The Models workflow can now build the required dependency layer from an
  operator-approved immutable worker image. The build context contains only a
  validated `requirements.lock`, uses pip `--require-hashes`, and labels the
  result with the driver-package hash; uploaded Python never enters the
  network-enabled build. The returned immutable local image ID and receipt are
  content-addressed and HMAC-signed before the image becomes eligible for the
  network-off conformance action. On restart, AudioEnv restores only receipts
  whose artifact hash, signature, metadata, and approved-base trust chain all
  verify. Tampered receipts are ignored. This build transport is fake-Docker tested; the current
  host still lacks the rootless CUDA Docker capability for a real build.
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
- Imported Harbor task collections now have a protected OCI validation worker
  and product action. The worker requires rootless Docker, builds without
  network by default, runs every oracle/no-op replay twice plus an explicit
  adversarial fixture in fresh network-off, non-root, read-only containers,
  hashes rather than returns private output, verifies state isolation and
  source/verifier immutability, and automatically signs and promotes only when
  all hard gates pass. Fake-container tests exercise the complete lifecycle;
  this Mac has no rootless Docker runtime, so real untrusted-container
  acceptance remains unrun and the endpoint fails closed here.
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
- Trainer registry, API preflight, and dashboard records expose release
  maturity instead of presenting both algorithms equally: GSPO is the
  `release_candidate`; GRPO is `experimental` until independently accepted on
  real CUDA hardware.
- Malformed structured actions and rejected tau2 tool calls no longer abort a
  costly rollout. They become traceable recovery observations with deterministic
  process penalties, including a stronger penalty for identical repeated
  failures; protected terminal verification is unchanged and private worker
  errors are not exposed to the policy.
- The immutable worker now binds all trainer and caller CUDA ordinals to the
  attested physical inventory and rejects missing or sub-80-GB devices before
  model loading, closing a case where a four-GPU host could pass VeRL preflight
  but fail when the separately allocated caller started.
- Checkpoint resume is now a product workflow rather than a hidden trainer
  option. The API creates an immutable lineage-bound child run after verifying
  the checkpoint artifact hash; GPU children additionally require matching
  successful worker reload evidence. The dashboard resumes a local run,
  preflights it, explicitly launches it, and shows continued step numbering. GPU resume
  drafts resolve preserved `global_step_*` directories and RunPod fails closed
  without its original persistent volume. Provider preparation rejects a
  different local workspace, RunPod network volume, or Modal volume. The run
  detail UI carries those GPU drafts through fresh configuration validation,
  provider review, and exact-name confirmation before it permits relaunch.
- The GPU run wizard now emits VeRL's actual `max_steps` setting. The previous
  generic `steps` field could be ignored and silently fall back to one step.
- GPU checkpoint files collected through local CUDA, RunPod, or Modal now bind
  to a matching persisted `checkpoint.saved` event only when the worker's
  post-training reload evaluation succeeded. The verification endpoint checks
  each file's content hash and presents the worker reload provenance; artifacts
  without matching evidence fail closed. Local tiny-policy checkpoints retain
  their independent load-and-deserialize verification path.
- The VeRL agent loop no longer supplies a precomputed reward score, which
  would cause upstream batching to bypass the configured scorer. Protected
  terminal/process results and the immutable verifier hash now flow through
  tool-extra fields, forcing VeRL's reward loop to call AudioEnv's hash-bound
  `compute_audioenv_score` before `rm_scores` are created.
- The Qwen worker now configures immutable held-out validation before training
  and at the final optimizer step, requires a saved `global_step_` checkpoint,
  launches a second validation-only VeRL process from that exact checkpoint,
  and rejects post-reload outcome drift. It exports protocol-hashed baseline,
  reloaded-trained, paired-comparison, and checkpoint-reload evidence. This is
  contract-tested locally; real results remain CUDA-gated.
- The pinned VeRL FSDP optimizer is now instrumented in each worker rank. Every
  step records the non-zero gradient names, declared trainable names, finite
  gradient norm, and before/after trainable-parameter hashes; training fails if
  anything outside `q_proj`/`k_proj`/`v_proj`/`o_proj` LoRA receives gradients,
  a rank is missing, or parameters do not change. The trainer controller also
  recomputes policy log-probabilities and fails before loss computation when
  masked rollout/training drift exceeds the configured tolerance. Both evidence
  summaries are exported, but real distributed evidence remains CUDA-gated.
- The Qwen rollout now selects vLLM-Omni's full Thinker → Talker → Code2Wav
  pipeline. A revision-pinned bridge retains the stage-0 policy tokens and
  log-probabilities plus the stage-2 waveform from the same generation request
  and fails closed if either is absent. This closes the prior architectural
  gap where Thinker-only rollout could not produce natively bound agent audio;
  CUDA execution of the bridge remains a release gate.
- The frozen Qwen caller now has a loopback-only service boundary with one
  model resident on its assigned GPU set and isolated conversation state for
  every rollout. Private goals remain inside the service, responses return
  native WAV bytes, and the client refuses non-loopback URLs. The GPU worker
  starts, health-checks, monitors, and stops this service separately from
  VeRL, with disjoint caller and trainer GPU allocations.
- A protected tau2 native-episode controller now routes the private scenario
  only to that caller service, passes only caller WAV and permitted tool
  results to the policy, executes strict structured tool/speech envelopes,
  records semantic turns for upstream evaluation, cleans up caller sessions,
  and takes terminal reward from tau2's official evaluator.
- The production Qwen/tau2 runtime factory now drives VeRL's live policy
  server with native caller audio, requires sampled-token log-probabilities,
  binds same-request Talker WAV to spoken actions, inserts tool/caller tokens
  with training mask zero, and fails on non-append-only policy lineage. A
  postprocessing bridge waits for VeRL's Qwen processor to calculate the real
  four-channel multimodal RoPE positions, then creates `TrajectoryV2` with
  exact audio-token spans and caller/agent audio lineage. This code remains
  CUDA-unverified.
- `audioenv tau2-dataset` now compiles the pinned worker's official public
  train or test membership into canonical, hash-bound VeRL JSONL plus a
  manifest. Rows carry the exact environment, verifier, model, checkpoint,
  policy, reward, seed, and split metadata required by the runtime without
  copying the private user scenario. The Qwen worker and trainer require
  different train and held-out validation files and reject accidental reuse.

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
  preserved instead of being overwritten by the aggregate `base` list. The
  explicitly gated `tau2-validate-reference` command replayed upstream
  reference actions for tasks `0` and `33`, covering both retail reward-basis
  combinations (`DB+NL_ASSERTION` and `DB`); τ²'s official evaluator returned
  `1.0` for both the full and environment rewards. The command returns only a
  redacted reward summary and is disabled in ordinary rollout workers.
- Declarative in-process environments now receive backend-issued promotion
  evidence only after a constrained subprocess independently reconstructs the
  environment and reproduces both its environment and verifier hashes. The
  worker uses a sanitized environment, bounded input and deadline, closed
  descriptors, and host-supported POSIX resource limits. Harbor candidates
  now produce equivalent signed evidence through the OCI worker when its
  rootless-Docker capability and all protected gates pass.
- `AudioEnvAgentLoop` validates policy version, masks, token/log-probability
  alignment, grouped rollouts, zero-variance rate, and VeRL-shaped V2 batch
  export. `AudioEnvVerlAgentLoop` now subclasses and registers against the
  pinned VeRL `AgentLoopBase` when loaded in the CUDA worker, converts an
  append-only `TrajectoryV2` into VeRL's exact prompt/response/mask/log-prob
  output, and masks caller/tool observation tokens to zero. It refuses an
  implicit scripted fallback. The frozen caller, tau2 episode controller,
  live-policy runtime factory, and processor-derived trajectory finalizer are
  implemented and locally contract-tested. The complete loop has not yet run
  inside the pinned CUDA worker.
- The VeRL manager's postprocessing hook now installs at runtime and converts
  native rollout drafts into persisted `TrajectoryV2` records using the
  processor-derived multimodal positions. A regression test verifies the
  manager method is truly replaced exactly once; previously only the
  standalone finalizer was tested and unreachable installation statements
  left the real hook inactive.
- Pinned-source inspection then showed that VeRL V1 actually queues rollouts
  through `AgentLoopWorkerTQ`, not the legacy manager. The hook now targets
  that concrete worker, derives multimodal positions with its own processor,
  finalizes before queue insertion, and is exercised by an executable fake-V1
  worker test. Invalid lineage therefore cannot enter the trainer queue.
- Multi-turn finalization now resolves caller-audio spans across the complete
  flattened prompt and masked response sequence. Later caller observations are
  mask-zero context, but remain traceably bound to their audio artifacts;
  feature spans that escape the policy sequence are rejected.
- A dependency-light end-to-end Qwen/τ² runtime contract test now drives the
  production rollout function through a protected task reset, frozen caller
  session, structured tool call, intermediate spoken turn, second caller-audio
  observation, terminal speech, official reward, mask-zero observation tokens,
  and caller cleanup. Model and service boundaries are faked; CUDA acceptance
  is still required for the real checkpoint.
- The Qwen rollout now consumes the actual `global_steps` value supplied by
  pinned VeRL V1, combines it with the immutable dataset policy lineage, and
  emits equal minimum/maximum actor-step bounds. This supplies VeRL's required
  staleness metadata and prevents online trajectories from retaining a false
  static base-policy version after optimizer updates.
- The immutable GPU image now runs `audioenv-verl-contract-check` after
  installing the pinned VeRL/VeRL-Omni packages. It verifies the exact agent
  output fields, async postprocessor hook, `audioenv_agent` registration,
  `pkg://` protected reward loader, protected component weighting, and `naive`
  reward-manager registration. Worker preflight repeats the check and records
  its evidence; any upstream API drift fails before model loading or training.
  The two pinned repositories were also inspected at their recorded commits,
  but the image-level command still awaits a real Docker/CUDA image build.
- That image-build command now forces native integration mode and verifies the
  active markers for Thinker/Talker output binding, trajectory finalization,
  LoRA-gradient auditing, and log-probability parity. Missing or drifted patch
  points fail the image build rather than surviving a shallow import check.
- The release image wrapper now verifies that the wheel metadata matches the
  dependency lock, sources every upstream build argument from that lock,
  attaches wheel/lock SHA-256 labels, and emits a versioned receipt containing
  the immutable local Docker image ID and all release pins. Contract tests use
  fake Docker and also reject wheel-version drift. A local image ID is not
  misrepresented as a pushed registry digest; both the real CUDA build and
  registry publication remain external gates.
- RunPod has REST v2 quote/provision/state/stop methods plus a concrete
  strict-host-key SSH/SCP data plane for staging, detached worker launch,
  reconnectable event replay, safe-boundary cancellation, and hash-verified
  artifact collection. These paths are locally transport-tested; the live API,
  Pod SSH endpoint, container image, and paid execution are not verified.
- Modal has a concrete optional SDK transport and a persisted dashboard/API
  workflow. Modal does not provide a live market-price quote through this
  integration, so its UI labels the configured rate as an estimate. An
  external paid Sandbox acceptance job has not run.

## Blocked

- Qwen3-Omni-30B-A3B native caller/agent execution and VeRL-Omni GSPO LoRA
  require approved Linux CUDA infrastructure, model weights, credentials, a
  live quote, maximum budget/runtime, and typed launch confirmation. GRPO has
  the same external prerequisites but remains an experimental second gate.
- The UI reports `Integration ready; GPU acceptance not run`.
- Native-audio learning lift, independent Qwen caller transfer, GPU utilization,
  and paid cost remain blocked until that run succeeds.

## Missing / out of scope

- Arbitrary repository execution remains unavailable. The implemented Harbor
  validation path is intentionally narrower: it accepts a Harbor bundle with
  a reference solution, protected verifier, and explicit adversarial fixture.
  Real rootless-Docker acceptance remains outstanding on a compatible host.
- Full duplex, interruption, telephony, Talker gradients, multi-tenancy,
  billing, Kubernetes, and additional cloud providers beyond RunPod and Modal
  are outside V1.

## Verification evidence

```text
uv run --extra dev --extra platform --extra train pytest -q
174 passed

uv run audioenv validate
389 passed, 0 failed

cd web && npm run build && npm run test:e2e
Vite production build passed
6 Playwright product-flow tests passed

uv build
audioenv_foundry-0.5.0 wheel and sdist built
```

The product-flow gates cover project creation, resource selection, preflight,
explicit launch, worker completion, real parameter updates, checkpoint reload,
two held-out evaluations, compatible paired comparison, artifact download
exposure, and SPA deep-link serving.
