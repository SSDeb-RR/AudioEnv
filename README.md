# AudioEnv Foundry

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
- Fictional retail and dual-control telecom reference tasks.
- Oracle, no-op, mutation, determinism, leakage, and parallel-isolation checks.
- Leakage-safe public export plus separately stored private verifier material.
- Local parallel rollouts and trainer-neutral JSONL export with tokens/logprobs.
- Optional OpenEnv, Verifiers, and Qwen2.5-Omni integration boundaries.

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

`NativeAudioModelAdapter` returns the environment action alongside token IDs,
log-probabilities, and metadata. `RolloutWorker` writes those values into the
trace and `export_trainer_jsonl` produces trainer-neutral JSONL.

`Qwen25OmniAdapter` accepts an injected generation callback. Model loading,
device placement, vLLM serving, LoRA updates, and the GRPO optimizer remain in
the lab's existing stack.

Install optional integrations only when needed:

```bash
uv sync --extra qwen
uv sync --extra openenv
uv sync --extra verifiers
```

The OpenEnv integration implements its `reset`, `step`, and `state` server-side
contract when `openenv-core` is installed. The Verifiers helper emits redacted
task rows and an in-process runtime; it never exposes verifier truth.

## Repository map

```text
src/audioenv/spec.py          canonical public contracts
src/audioenv/runtime.py       isolated episode execution
src/audioenv/rewards.py       process and terminal verification
src/audioenv/validation.py    pre-training integrity checks
src/audioenv/rollout.py       parallel rollouts and trainer export
src/audioenv/adapters/        benchmark conversion SDK
src/audioenv/integrations/    optional OpenEnv and Verifiers bridges
src/audioenv/models/          native-audio model adapter boundary
src/audioenv/reference.py     retail and telecom reference package
```

## Release gates not claimed by this repository

The following require external resources and are deliberately not represented
as completed:

- a Qwen2.5-Omni-7B baseline and LoRA/GRPO training run;
- a ten-point held-out pass@1 improvement;
- transfer to an independent native-audio caller;
- GPU throughput and cost measurements;
- design-partner interviews, letters of intent, and a paid private pilot;
- benchmark redistribution approval.

The executable foundation is ready for those experiments, but none should be
claimed until its corresponding evidence exists.

