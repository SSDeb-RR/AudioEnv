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

## Training boundary

The trainer consumes validated `TrajectoryV1` records, not environment objects.
Trainers declare their algorithms, model families, trainable modules,
distributed backends, checkpoint format, and resume support. Unsupported
configurations fail before queueing.

The local trainer is a real Torch policy-gradient smoke test. The production
adapter delegates Thinker LoRA/GSPO to pinned VeRL-Omni. Environment and reward
code contain no optimizer-specific behavior.

## Persistence and lifecycle

Relational storage contains registries, immutable experiment snapshots, jobs,
events, metrics, and artifact references. Large payloads use the artifact
store. State transitions are validated as:

```text
Draft -> Validating -> Queued -> Running -> Evaluating -> Completed
                         \          \             \
                          -> Failed / Cancelled <-
```

The local provider uses subprocess workers and persisted heartbeats. An
interrupted non-resumable job becomes failed; a resumable trainer may enqueue a
new attempt pointing to a checkpoint.

