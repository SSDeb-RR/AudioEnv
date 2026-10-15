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

## Evidence recorded on 2026-10-15

| Gate | Status | Evidence |
| --- | --- | --- |
| V0 and V1 Python suite | PASS | 41 tests passed with platform, training, EnvHarness, and Qwen extras installed |
| Built-in environment validation | PASS | 18 checks passed, including audio-profile and wrong-order checks |
| Python package build | PASS | `audioenv_foundry-0.2.0` wheel and sdist built |
| Dashboard dependency audit | PASS | `npm audit` reported zero vulnerabilities |
| Dashboard production build | PASS | TypeScript and Vite build completed |
| One-command local startup | PASS | `audioenv dev` served API health and the Vite application on loopback; job workers launch only after explicit run confirmation |
| Dashboard visual navigation | PASS, MANUAL | Overview, Models, Environments, Training Runs, and persisted metric detail were inspected in the local browser; automated Playwright coverage remains open |
| Persisted local workflow | PASS | 40-step subprocess run completed with 82 persisted metrics, checkpoint artifact, and evaluation record |
| CPU parameter update | PASS | Initial and final parameter hashes differ; checkpoint reload reproduces the trained result |
| CPU fixture comparison | PASS, LIMITED | Baseline `0.0`, trained `1.0`; mechanics-only two-task fixture, not product learning evidence |
| Official EnvHarness type integration | PASS | Exact bridge test passed against pinned upstream commit |
| Qwen3-Omni dependency probe | PASS | Dependencies import; probe correctly blocks on missing CUDA and local checkpoint without downloading |
| Real Qwen native-audio episode | BLOCKED | No CUDA device or checkpoint on the audit host |
| VeRL-Omni GSPO update | BLOCKED | Requires separately approved multi-GPU worker |
| Independent caller transfer | NOT RUN | Depends on the real native-audio training run |
| Ten-point product learning lift | NOT RUN | CPU fixture result is explicitly ineligible |
| OCI coding sandbox | BLOCKED | No Docker-compatible runtime is installed |

