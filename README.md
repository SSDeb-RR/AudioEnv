# AudioEnv Foundry

> **V1 status: local product acceptance passes; native GPU gate blocked.** The
> one-command web product performs a real non-oracle CPU policy update, reloads
> its checkpoint, and produces a protocol-safe 24-sample held-out comparison.
> Qwen3-Omni/VeRL-Omni remains unverified on this non-CUDA host. See
> [`docs/V1_IMPLEMENTATION_STATUS.md`](docs/V1_IMPLEMENTATION_STATUS.md).

AudioEnv Foundry is an Apache-2.0, trainer-neutral foundation for turning
structured agent benchmarks into deterministic, turn-based, audio-native RL
environments.

The runtime keeps the conversational channel as audio, keeps tool traffic
structured, isolates episode state, and emits process plus terminal rewards.
It includes one local REINFORCE optimizer and Qwen3-Omni GSPO/GRPO trainer
plugins, but does **not** invent a new distributed optimizer, telephony,
full-duplex streaming, or closed-model RL.

## What works now

- Canonical environment, task, audio, action, reward, and trace contracts.
- Isolated `reset` / `step` runtime with seeded audio generation boundaries.
- Exact terminal verification of state, tools, communication, and safety.
- Milestone-only process rewards; unrelated successful calls earn nothing.
- Penalties for invalid, repeated, and unauthorized calls.
- JSON benchmark adapter with an explicit audited tool-handler registry.
- Frozen 48-task retail/telecom corpus with disjoint 32/8/8 splits.
- Oracle, no-op, mutation, determinism, leakage, and parallel-isolation checks.
- Leakage-safe public export plus separately stored private verifier material.
- Local parallel rollouts and trainer-neutral JSONL export with tokens/logprobs.
- Optional OpenEnv, Verifiers, and Qwen2.5-Omni integration boundaries.
- Versioned V1 environment, generation, trajectory, model-capability, trainer,
  and resource-estimate contracts with alignment validation.
- Declarative custom-environment authoring with separately stored protected
  verifier artifacts.
- EnvHarness-compatible bridge plus safe `Setup` and restricted `Rules`
  adaptations that retain the original evaluator.
- SQLite/SQLAlchemy experiment persistence, enforced lifecycle transitions,
  metrics, evaluations, artifacts, cancellation, worker heartbeats, and
  abandoned-worker recovery.
- FastAPI service and React/TypeScript dashboard driven by persisted records.
- A genuine audio-conditioned Torch REINFORCE workflow with no oracle replay,
  train-only optimization, checkpoint reload, and 24-sample held-out comparison.
- Leakage-safe local and native VeRL training trajectories streamed to
  hash-bound JSONL, with a run-scoped API and dashboard view of public actions,
  rewards, state hashes, available timing, and terminal outcomes.
- A dynamic harness registry for environments, model families, callers,
  trainers, and execution providers, including Python entry-point discovery.
- A repository/benchmark Foundry that safely inventories source without
  importing it, compiles recognized declarative tasks, and emits reviewable
  OCI coding-environment scaffolds when repository tests are detected.
- Optional Codex-assisted environment authoring from a bounded, redacted
  source digest. Its schema-constrained output remains untrusted, validated,
  and human-review-required; Codex is never invoked during startup or tests.
- Omni model onboarding that selects an architecture-family driver or emits a
  conformance scaffold, plus capability-derived caller and trainer choices.
- Guarded upload of implemented Omni-driver source packages. Uploads are
  statically inspected, content-addressed, and kept unavailable until isolated
  executable conformance succeeds; uploaded Python is never imported by the API.
- Rootless OCI conformance for uploaded drivers using an operator-approved
  immutable image (`AUDIOENV_MODEL_CONFORMANCE_IMAGES`),
  network-off/read-only execution, optional read-only model weights, all ten
  model gates, and content-addressed evidence. Only a complete report promotes
  the model to `conformant`; this host has not run the GPU container gate.
- A controlled dependency-image builder derives from that approved base and
  installs only hash-locked requirements. Driver source is excluded from the
  network-enabled build context and runs later in the network-off sandbox.
  Signed, content-addressed receipts preserve only verified image trust across
  application restarts.
- Qwen3-Omni Transformers and VeRL-Omni GSPO/LoRA adapters with truthful
  capability and hardware preflight failures.
- Pinned Repo2RLEnv-to-Harbor import in an isolated container, with protected
  verifier storage and automatic promotion only after every integrity gate.
  A rootless-Docker Harbor validator runs fresh oracle, no-op, and explicit
  adversarial trials without returning private verifier output; it fails
  closed when that runtime or a required fixture is unavailable.
- `TrajectoryV2`, a model-driven caller/agent rollout worker, and distinct
  VeRL-Omni v0.2 GSPO and GRPO Thinker-LoRA plugins. These paths are code-ready
  but remain GPU-unverified.
- A provider-neutral `audioenv-gpu-worker` bundle/event/artifact contract used
  by the verified local training backend and the CUDA-gated Qwen backend.
- An executable pinned-VeRL API audit that verifies agent-loop registration,
  rollout fields, protected reward loading, and reward-manager registration
  while building or attesting the GPU worker runtime.

The bundled WAV renderer generates deterministic audio fixtures for tests. It
is not a speech synthesizer. A real deployment must inject a native-audio user
model through the `AudioRenderer` protocol. Transcripts may be recorded as an
observability sidecar but are never returned by `AudioObservation.agent_view()`.

## Quick start

```bash
uv sync --extra dev
uv run pytest
uv run audioenv validate
uv run audioenv demo --domain retail --trajectory oracle --output outputs/retail-trace.json
uv run audioenv demo --domain telecom --trajectory wrong
uv run audioenv export outputs/reference-environment

# V1 production app and real CPU optimizer proof
uv sync --extra dev --extra platform --extra train
cd web && npm install && npm run build && cd ..
uv run audioenv app --data-dir .audioenv

# Open http://127.0.0.1:8000. The wheel includes the compiled SPA; Node is
# needed only to rebuild it. `audioenv dev` remains the hot-reload command.

# In the dashboard, open Foundry and submit a local path or an HTTPS GitHub
# repository. Repository code is inspected but never imported or executed.
# Choose Codex authoring explicitly only when an authenticated Codex CLI is
# available. Repo2RLEnv mode additionally requires Docker and its pinned image;
# deterministic inspection is the credential-free default.

# This does not download weights; it reports why Qwen is or is not available.
uv run --extra qwen audioenv qwen-probe
```

An oracle trajectory should end with `terminal_pass: true`; no-op and wrong-slot
trajectories should end with `terminal_pass: false`.

## Runtime contract

```python
from audioenv.reference import build_reference_package
from audioenv.runtime import AudioEnvironment
from audioenv.spec import ToolCallAction

env = AudioEnvironment(build_reference_package())
observation = env.reset("retail-refund-001", seed=7)

# `observation.audio` is a WAV byte string. No source transcript is exposed.
observation = env.step(
    ToolCallAction("lookup_order", {"order_id": "A-1042"})
)
```

The contract is:

```text
reset(task_id, seed) -> AudioObservation
step(SpeakAction | ToolCallAction) -> AudioObservation
```

`state_snapshot` is evaluator-facing and must not be added to an agent context.
`public_state` contains only episode metadata and a state hash.

## Converting a benchmark

Implement `BenchmarkAdapter`, or use `JsonBenchmarkAdapter` when the source can
be normalized into `audioenv/v1`. An authoring manifest supplies:

- environment policies, tools, profiles, and episode limits;
- task initial state and expected state;
- initial and subsequent private caller utterances;
- exact required tool milestones and communication assertions;
- forbidden operations and validation-only oracle actions.

Executable tool handlers are not loaded from JSON. They must be supplied as an
audited Python registry:

```python
from audioenv.adapters import JsonBenchmarkAdapter

adapter = JsonBenchmarkAdapter({"lookup_order": lookup_order})
package = adapter.convert(path_to_authoring_manifest)
```

This makes code execution an explicit trust decision. `audioenv export` writes
public task metadata to `environment.json` and verifier truth to
`.audioenv-private/verifier.json`; oracle actions are never exported.

## Reward semantics

Terminal success is conjunctive:

```text
expected state AND required milestones AND required communication AND safety
```

Process rewards are issued only when an exact required milestone is reached.
A valid but irrelevant call receives no positive reward. Failed calls receive
`-0.1`, repeated failures receive an additional `-0.1`, and blocked forbidden
mutations receive `-0.5`. The terminal outcome remains a separate `0` or `1`
event so trainers can choose whether and how to combine shaping.

## Model and trainer integration

The legacy `NativeAudioModelAdapter` remains compatible with V0 callback-based
integrations. V1 adds `GenerationRecord` and `TrajectoryV1`: action tokens,
response masks, multimodal positions, old-policy log-probabilities, audio
artifacts, policy versions, and episode boundaries are validated together.
`TrajectoryV2` adds explicit multimodal feature spans, audio lineage, separate
terminal/process reward provenance, and grouped-rollout validation.

Pinned tau2 tasks can be compiled into public, hash-bound VeRL inputs without
including the private caller scenario:

```bash
audioenv tau2-dataset \
  --checkout /path/to/pinned/tau2 \
  --output train.jsonl \
  --split train \
  --experiment-id EXPERIMENT_ID \
  --environment-hash ENVIRONMENT_SHA256 \
  --verifier-hash VERIFIER_SHA256 \
  --policy-version POLICY_VERSION
```

Generate the held-out `test` file separately. Qwen training preflight rejects
configurations that reuse the training file as validation data.

`Qwen3OmniAgentAdapter` and `Qwen3OmniCallerAdapter` use the upstream
Transformers processor/model interfaces, preserve semantic-token/Talker-audio
binding, and parse constrained structured tool envelopes. Their `probe()` does
not download weights and reports missing CUDA, checkpoint, dependency, and
capacity requirements explicitly. The CPU `SmallAudioToolPolicyTrainer` is a
real audio-conditioned platform proof. `VerlOmniGSPOTrainer` and
`VerlOmniGRPOTrainer` are distinct strict launch plugins for pinned VeRL-Omni
v0.2 and are not represented as having run on this host.

Install optional integrations only when needed:

```bash
uv sync --extra qwen
uv sync --extra train
uv sync --extra envharness
```

The OpenEnv integration implements its `reset`, `step`, and `state` server-side
contract when `openenv-core` is installed. The Verifiers helper emits redacted
task rows and an in-process runtime; it never exposes verifier truth.

## Repository map

```text
src/audioenv/contracts.py     versioned environment, trajectory V1/V2, and GPU contracts
src/audioenv/runtime.py       isolated episode execution and action binding
src/audioenv/rewards.py       process and terminal verification
src/audioenv/validation.py    pre-training integrity checks
src/audioenv/rollout_v2.py    model-driven caller/agent trajectory collection
src/audioenv/adapters/        benchmark conversion SDK and tau2 importer
src/audioenv/integrations/    optional ecosystem compatibility bridges
src/audioenv/models/          scripted and Qwen3-Omni adapter implementations
src/audioenv/training/        CPU trainer and VeRL-Omni launch boundary
src/audioenv/platform/        API, persistence, worker, execution, and exports
web/                          React/TypeScript operator dashboard
docker/gpu-worker/            immutable-base GPU worker image definition
```

## Release gates not claimed by this repository

The following require external resources and are deliberately not represented
as completed:

- Qwen3-Omni-30B-A3B baseline and Thinker LoRA GSPO and GRPO training runs;
- a ten-point held-out pass@1 improvement;
- transfer to an independent native-audio caller;
- GPU throughput and cost measurements;
- design-partner interviews, letters of intent, and a paid private pilot;
- benchmark redistribution approval.

The executable foundation is ready for those experiments, but none should be
claimed until its corresponding evidence exists.

## Dashboard

The dashboard exposes Overview, Projects, Models, Benchmarks & Tasks, Environments,
Training Runs, Evaluations, Artifacts, and Settings. It reads only backend
records. Missing cost, GPU, evaluation, or model-probe data is displayed as
unavailable rather than replaced with demo values.

Production frontend build:

```bash
cd web
npm install
npm audit
npm run build
```

## V1 documents

- [`docs/V1_AUDIT.md`](docs/V1_AUDIT.md)
- [`docs/V1_ARCHITECTURE.md`](docs/V1_ARCHITECTURE.md)
- [`docs/V1_PLAN.md`](docs/V1_PLAN.md)
- [`docs/V1_ACCEPTANCE_CRITERIA.md`](docs/V1_ACCEPTANCE_CRITERIA.md)
- [`docs/V1_IMPLEMENTATION_STATUS.md`](docs/V1_IMPLEMENTATION_STATUS.md)

