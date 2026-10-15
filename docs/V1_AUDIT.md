# AudioEnv V1 implementation audit

Audit date: 2026-10-15  
AudioEnv commit: `24b7d033779e2adea95df4777235214121ad8af7`  
EnvHarness commit inspected: `fab7d57441f06b75c73a900e04561d4d7600f361`  
VeRL-Omni commit inspected: `f31a6116119fd62a89554987f51ae279a567d66b`

## Evidence summary

The V0 baseline passed all 20 tests, all 14 `audioenv validate` checks, and a
wheel/sdist build on Apple Silicon macOS. No NVIDIA device, Docker-compatible
runtime, or ffmpeg executable is installed on the audit machine.

## What works

- The dependency-light SDK defines environment, task, action, audio-profile,
  reward, observation, and trace records.
- `AudioEnvironment` deep-copies task state, implements synchronous reset/step,
  executes registered structured tools, and produces process and terminal
  rewards.
- The two fictional tasks exercise a conventional retail flow and a
  dual-control telecom flow.
- Public export separates verifier data into `.audioenv-private` and excludes
  authoring-only oracle actions.
- Validation covers oracle success, no-op rejection, argument mutation,
  deterministic replay, basic leakage, and parallel state isolation.
- Rollouts retain generated token IDs and log-probabilities supplied by a model
  adapter and can be exported as JSONL.

## Partial implementations

- `Qwen25OmniAdapter` is a callback wrapper. It neither loads a checkpoint nor
  calls an audio processor or Talker.
- The built-in WAV renderer generates short deterministic tones. It is useful
  for contract tests but is not speech, and profile labels such as codec and
  noise do not currently transform the waveform.
- The JSON benchmark adapter only accepts an already-normalized AudioEnv
  manifest and a trusted in-process handler registry.
- OpenEnv and Verifiers modules express useful boundaries but were not
  integration-tested against their upstream packages.
- Traces contain optional token IDs and log-probabilities but not the masks,
  policy versions, multimodal positions, artifact lineage, and batch grouping
  needed to validate a policy update.

## Interfaces and placeholders

- `NativeAudioModelAdapter` declares only `act`; it has no capability probe,
  checkpoint, trainability, or processor contract.
- `RolloutWorker` serializes episodes but does not calculate advantages or
  update parameters.
- Caller turns are static private strings rendered as tones. There is no caller
  model or conversation-state adapter.
- Audio profile accent, codec, speaking rate, prosody, and SNR fields are
  metadata rather than verified processing operations.

## Missing components

- Experiment persistence, lifecycle state machine, worker recovery,
  cancellation, metrics, checkpoint registry, and evaluation comparison.
- Backend API and dashboard.
- Real model loading, native speech generation, a model-driven caller, and GPU
  preflight checks.
- Trainer integration, advantage calculation, optimizer state, checkpoint
  reload, and parameter-delta verification.
- A protected versioned environment IR, custom authoring workflow, real public
  benchmark importer, EnvHarness adaptation proof, and coding sandbox.

## Architectural problems

1. `SpeakAction.semantic_text` is supplied directly by the actor and is used by
   the reward tracker. It is not cryptographically or structurally bound to the
   generated waveform.
2. Private verifier data is separated at export time but remains available on
   the same in-process task object as policy-visible data.
3. Terminal state comparison is subset-based, so unrelated destructive state
   mutations require explicit additional safety rules.
4. Tool handlers are trusted in-process Python without timeout or isolation.
5. Initial caller audio is not represented as a first-class trace artifact.
6. Runtime failures outside a small exception allowlist terminate the episode
   without a normalized failure record.
7. The two-task fixture is insufficient for a learning or generalization
   claim.

## Retain, refactor, replace

- Retain the synchronous runtime, registered-handler trust decision, reference
  tasks, deterministic rewards, and current V0 public methods.
- Refactor serialization around versioned Pydantic boundary models; keep the
  existing dataclasses as the compiled runtime form.
- Refactor model, trace, and rollout contracts to carry explicit capabilities,
  generation provenance, masks, policy versions, and audio artifacts.
- Replace arbitrary speech semantic text with adapter-created generation
  records.
- Replace fake acoustic labels with implemented transformations or explicit
  validation errors.
- Do not replace the environment runtime with EnvHarness or the trainer with a
  new optimizer.

## Smallest genuine training path

The smallest locally verifiable path is a Torch policy that consumes a feature
from the actual reset audio, samples an environment strategy, receives the
environment terminal reward, performs a policy-gradient update, saves a
checkpoint, reloads it, and compares baseline and trained success. This proves
platform mechanics but is not a native-speech model claim.

The release path is Qwen3-Omni-30B-A3B-Instruct with native audio input and a
frozen Talker, plus Thinker LoRA/GSPO through VeRL-Omni. That path remains a
manual GPU release gate because the audit host cannot execute it.

