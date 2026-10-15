# AudioEnv V1 architecture

## Components

```text
React dashboard ──HTTP/SSE── FastAPI service ──SQLAlchemy── SQLite/PostgreSQL
                              │       │
                              │       └── content-addressed artifact store
                              │
                              └── execution provider ── local/GPU worker
                                                       │
                    protected verifier ── environment runtime
                                            │       │
                         caller audio ──────┘       └── model action
                                                        │
                                            trajectory ─┴─ trainer/evaluator
```

The SDK stays usable without the platform server. Platform, training, and Qwen
dependencies are optional extras and are imported lazily.

## Harness Foundry

`HarnessRegistry` is the common discovery surface for environment, model,
caller, trainer, and execution-provider plugins. Built-ins and Python entry
points publish the same capability descriptors. The compatibility resolver
intersects environment requirements, model capabilities, caller requirements,
trainer requirements, and runtime availability before the UI offers a choice.

`EnvironmentFoundry` accepts a local source or HTTPS GitHub repository. It
performs bounded, non-executing inventory. A declarative `audioenv-task.json`
is compiled through the deterministic author and fully validated. An
`audioenv/v1` benchmark is inspected and can execute only when its tool
handlers are independently audited. A repository with tests generates a
protected coding-environment and OCI-sandbox scaffold, but no repository code
is run on the host.

The optional `CodexEnvironmentAuthor` receives only a 64 KB allowlisted,
secret-redacted text digest rather than source filesystem access. It runs only
after an explicit request, in an ephemeral read-only Codex session with a
`CustomTaskInput` output schema. The candidate is compiled by the constrained
deterministic author, independently validated, and stored as
`generated_needs_review`; it cannot redefine a validated verifier or become a
selectable training environment without review.

`ModelFoundry` reads a local model configuration when present, selects a
reusable architecture-family driver, and records conformance gates. Unknown
families receive a content-addressed driver scaffold and remain
`needs_review`. Model identifiers never imply that weights were downloaded or
loaded.

Repo2RLEnv generation is an explicit Foundry mode pinned to commit
`e5e6a93a6d7947a012a361b68457a982cc85fc9e`. It runs behind the restricted
`SandboxExecutor` for its `pr_diff` path, produces Harbor tasks, and is
imported without putting private test scripts into public task records.
Docker-dependent Repo2RLEnv pipelines remain unavailable until a sibling
sandbox executor exists; the host Docker socket is never mounted. Promotion is computed only when
sandbox, oracle, negative, determinism, isolation, leakage, reward-integrity,
and verifier-immutability gates all pass.

## Trust boundaries

- Model workers receive public task material, native caller audio, permitted
  tool results, and generation configuration. They do not receive hidden facts,
  expected state, verifier code, oracle actions, or observability transcripts.
- Environment workers own tool execution and protected verification. Verifier
  artifacts are content-hashed and immutable for an experiment.
- Semantic tokens and output audio enter the runtime through a trusted model
  adapter's single `GenerationRecord`; arbitrary actors cannot assert a separate
  reward-bearing transcript.
- Generated environment content is data, not executable Python. Optional LLM
  authors may create draft data but cannot modify a trusted verifier.
- Repository code is never executed on the host. Its environment remains
  unavailable until an OCI provider passes the sandbox capability check.
- Secrets are referenced by name and never serialized into snapshots, traces,
  observations, or exports.

## Environment construction

`EnvironmentPackageV1` is the portable boundary format. It records provenance,
capabilities, public task metadata, a protected verifier reference, audio
profiles, authoring status, validation evidence, and an immutable content hash.
The compiler validates it and produces the existing `EnvironmentPackage` used
by `AudioEnvironment`.

Custom authoring uses a deterministic scaffold by default. An optional author
provider may propose the same IR, but generated packages remain
`generated_needs_review` until separately validated and approved.

The first real benchmark adapter targets a pinned τ² retail release. It lives
in an isolated Python 3.12 worker to avoid coupling τ² dependencies and Python
constraints to the AudioEnv SDK.

## EnvHarness decision

AudioEnv implements a compatibility bridge for the seven-method
`ActionableEnv` contract and native declarative equivalents of `Setup` and a
restricted `Rules`. The protected AudioEnv verifier remains authoritative.
EnvHarness is an optional, pinned integration rather than a core dependency.
`Link` and arbitrary generated Python rules are deferred.

## Native-audio rollout and training boundary

`OmniModelDriver` is the architecture-family boundary. Qwen3-Omni is the
certified reference implementation; generated drivers cannot register before
native-audio, structured-action, alignment, gradient, update, and reload
conformance passes. Caller and agent histories are isolated.

The model-driven runner emits `TrajectoryV2`, binding audio hashes,
multimodal spans, prompt/generated tokens, response masks, rollout
log-probabilities, policy versions, actions, rewards, and episode boundaries.
Only agent Thinker action tokens receive response mask `1`.

The trainer consumes validated trajectories, not environment objects.
Trainers declare their algorithms, model families, trainable modules,
distributed backends, checkpoint format, and resume support. Unsupported
configurations fail before queueing.

The local trainer is a real Torch policy-gradient smoke test. Separate GSPO and
GRPO plugins delegate Qwen3 Thinker LoRA to VeRL-Omni v0.2 pinned at
`49dedb6ec0691588494b629fc1b610a86612966b`; both use grouped GRPO outcome
advantages. Talker, codec, audio tower, caller, and reference policy remain
frozen. Environment and reward code contain no optimizer-specific behavior.

## Persistence and lifecycle

Relational storage contains projects, registries, immutable experiment hashes
and preflight snapshots, jobs, events, worker logs, metrics, evaluation
protocols/samples, paired comparisons, split manifests, external resources,
and artifact references. Large payloads use the artifact store. State
transitions are validated as:

```text
Draft -> Validating -> Queued -> Running -> Evaluating -> Completed
                         \          \             \
                          -> Failed / Cancelled <-
```

The local provider uses subprocess workers and persisted heartbeats. An
interrupted non-resumable job becomes failed; a resumable trainer may enqueue a
new attempt pointing to a checkpoint.

The production entrypoint is `audioenv app --data-dir .audioenv`. It applies
packaged Alembic migrations, recovers expired worker leases, and serves the
compiled SPA and API from one loopback process. Local training still runs in a
separate subprocess and starts only after a persisted preflight and explicit
launch action.

## Local acceptance policy

Reference V2 exposes an explicit finite action-candidate surface separate from
its protected verifier/oracle fixtures. `ActionCandidateBuilder` can read only
that surface and public tool names. The small policy scores candidates using
WAV features and structural action features, samples a candidate with
REINFORCE, and executes its individual actions through the normal runtime.
Training iterates only the train split. Evaluation uses the eight test tasks
under three frozen acoustic profiles after checkpoint reload.

An evaluation protocol hashes environment, verifier, split manifest, caller,
audio pipeline, sampling, checkpoint, and seeds. Comparison omits only the
checkpoint identity when checking compatibility; all other differences reject
the comparison.

