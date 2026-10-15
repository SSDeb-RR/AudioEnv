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
It does **not** implement an optimizer, telephony, full-duplex streaming, or
closed-model RL.

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
- Qwen3-Omni Transformers and VeRL-Omni GSPO/LoRA adapters with truthful
  capability and hardware preflight failures.

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

`Qwen3OmniAgentAdapter` and `Qwen3OmniCallerAdapter` use the upstream
Transformers processor/model interfaces, preserve semantic-token/Talker-audio
binding, and parse constrained structured tool envelopes. Their `probe()` does
not download weights and reports missing CUDA, checkpoint, dependency, and
capacity requirements explicitly. The CPU `SmallAudioToolPolicyTrainer` is a
real audio-conditioned platform proof; `VerlOmniGSPOTrainer` is a strict pinned launch
boundary for the external CUDA trainer and is not represented as having run on
this host.

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
src/audioenv/contracts.py     versioned V1 IR, trajectory, and capabilities
src/audioenv/runtime.py       isolated episode execution and action binding
src/audioenv/rewards.py       process and terminal verification
src/audioenv/validation.py    pre-training integrity checks
src/audioenv/rollout_v1.py    validated V1 rollout collection
src/audioenv/adapters/        benchmark conversion SDK and tau2 importer
src/audioenv/integrations/    optional ecosystem compatibility bridges
src/audioenv/models/          scripted and Qwen3-Omni adapter implementations
src/audioenv/training/        CPU trainer and VeRL-Omni launch boundary
src/audioenv/platform/        API, persistence, worker, execution, and exports
web/                          React/TypeScript operator dashboard
```

## Release gates not claimed by this repository

The following require external resources and are deliberately not represented
as completed:

- a Qwen3-Omni-30B-A3B baseline and Thinker LoRA/GSPO training run;
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

