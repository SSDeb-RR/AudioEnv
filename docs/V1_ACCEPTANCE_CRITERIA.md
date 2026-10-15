# AudioEnv V1 acceptance criteria

Status values are `PASS`, `FAIL`, `BLOCKED`, and `NOT RUN`. Evidence must name
the command, revision, hardware, and artifact path.

## Local automated gate

- All V0 tests remain green.
- V1 schemas reject invalid hashes, split changes, non-finite log-probabilities,
  mask mismatches, missing episode boundaries, and policy-version drift.
- Environment tests cover oracle, no-op, wrong arguments, wrong order,
  forbidden tools, unexpected mutations, replay, state isolation, and leakage.
- Lifecycle tests cover every valid and invalid transition, cancellation,
  atomic job claim, and abandoned-worker recovery.
- The CPU training test proves a non-zero parameter change, finite loss,
  checkpoint save/reload parity, and a baseline-versus-trained evaluation.
- API tests use a temporary database and artifact root.
- Dashboard tests use backend fixtures created through the public API and show
  unavailable metrics honestly.
- Public export contains no hidden facts, oracle actions, verifier content, or
  secret values.

## GPU release gate

- Qwen3-Omni loads at a recorded immutable revision.
- A frozen caller and trainable agent exchange native audio over multiple turns.
- The agent produces a structured tool call and a Talker waveform.
- The trajectory passes multimodal token/mask/log-probability validation.
- GSPO updates only declared Thinker LoRA parameters.
- The saved adapter reloads with equivalent inference.
- Baseline and trained checkpoints use identical protected held-out manifests.
- Observability ASR is absent from model inputs and reward inputs.

## Product-learning gate

- Held-out pass@1 improves by at least 10 absolute percentage points.
- Improvement transfers to the independent caller/audio condition.
- Repeated failed calls and unauthorized mutations do not increase.
- Speech-quality and safety monitoring show no material regression.
- Local replay, trainer export, and evaluator produce reward parity.

Functional platform completion without the GPU and learning gates must be
reported as a preview, not AudioEnv V1 complete.

## Evidence recorded on 2026-10-16

| Gate | Status | Evidence |
| --- | --- | --- |
| V0 and V1 Python suite | PASS | 97 tests passed, including executable Omni conformance, same-request Thinker/Talker binding, frozen-caller session isolation and loopback transport, protected tau2 episode routing, native VeRL rollout masking, processor-derived trajectory finalization, disjoint GPU allocation, multi-channel audio RoPE preservation, isolated declarative validation, signed promotion evidence, caller-private tau2 evaluation, and the guarded Modal provider contract |
| Built-in environment validation | PASS | 389 checks passed across all 48 tasks; 0 failed, including per-episode state isolation |
| Python package build | PASS | `audioenv_foundry-0.5.0` wheel and sdist built; wheel inventory includes SPA, migrations, model adapters, and harness foundry |
| Dynamic environment harness | PASS, LIMITED | Declarative repository tasks compile and validate; arbitrary coding repositories produce a non-executable OCI scaffold pending sandbox review |
| Dynamic model harness | PASS, LIMITED | Unknown families receive a downloadable review-gated Python SDK; a fixture passes optimizer/reload/determinism conformance, while real Qwen conformance remains GPU-gated |
| Dynamic trainer/caller selection | PASS | API and dashboard options are capability-resolved; blocked backends remain visible with reasons |
| Dashboard dependency audit | PASS | `npm audit` reported zero vulnerabilities |
| Dashboard production build | PASS | TypeScript and Vite build completed |
| One-command local startup | PASS | `audioenv app --data-dir ...` applied migrations and served bundled SPA plus API from an isolated wheel install without Node |
| Dashboard product flow | PASS | Playwright covers project-to-comparison execution plus repository analysis and Omni model harness onboarding |
| Codex authoring boundary | PASS | Fake-runner test proves read-only/schema-constrained invocation, secret redaction, deterministic validation, and review-gated output without making a paid call |
| Persisted local workflow | PASS | 30-step subprocess run completed with 92 persisted metrics, checkpoint/manifest artifacts, two evaluations, and one paired comparison |
| CPU parameter update | PASS | Initial and final parameter hashes differ; checkpoint reload reproduces the trained result |
| Local held-out comparison | PASS, LIMITED | Baseline `0.0`, trained `1.0` over 24 samples from the immutable test split; audio-conditioned platform proof, not a Qwen result |
| Official EnvHarness type integration | PASS | Exact bridge test passed against pinned upstream commit |
| Pinned tau2 retail worker | PASS, LIMITED | Live Python 3.12 worker reported 114 tasks, reset a real task, exposed 16 public tool schemas, isolated caller-private instructions from public reset, and completed the protected native episode controller through the official full evaluator for a terminal no-op with reward 0.0; gold parity across all reward bases remains pending |
| Backend promotion evidence | PASS, LIMITED | API rejects client gate booleans, verifies protected artifact hashes, signs evidence, detects tampering, and binds promotion to the current environment hash; OCI repository validation remains pending |
| Isolated declarative validator | PASS | Subprocess reconstructs the custom environment under bounded input/time/resources and registration requires exact environment/verifier hash parity |
| Qwen3-Omni dependency probe | PASS | Dependencies import; probe correctly blocks on missing CUDA and local checkpoint without downloading |
| Real Qwen native-audio episode | BLOCKED | No CUDA device or checkpoint on the audit host |
| VeRL-Omni launch and reward contract | PASS, LIMITED | Launcher matches pinned V1 `main_omni` recipe; frozen caller, tau2 protected episode controller, live-policy runtime, hash-bound reward, native multimodal handoff, and postprocessor-derived `TrajectoryV2` are contract-tested; CUDA execution remains incomplete |
| VeRL-Omni GSPO and GRPO update | BLOCKED | Each algorithm still requires a separately approved multi-GPU worker and real agent-loop acceptance |
| Modal execution-provider contract | PASS, LIMITED | Mock transport verifies immutable staging, fresh quote and typed confirmation, gVisor request, event replay, budget stop, cancellation, reconnect, and hash-checked artifact collection; live SDK and paid Sandbox run not executed |
| Independent caller transfer | NOT RUN | Depends on the real native-audio training run |
| Ten-point native-model learning lift | NOT RUN | Local proof exceeds the threshold but is explicitly ineligible as native Qwen evidence |
| OCI coding sandbox | BLOCKED | No Docker-compatible runtime is installed |

