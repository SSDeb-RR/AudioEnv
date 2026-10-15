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
| V0 and V1 Python suite | PASS | 46 tests passed, including the API-to-worker product acceptance flow |
| Built-in environment validation | PASS | 341 checks passed across all 48 tasks; 0 failed |
| Python package build | PASS | `audioenv_foundry-0.3.0` wheel and sdist built; wheel inventory includes SPA, migrations, and model adapters |
| Dashboard dependency audit | PASS | `npm audit` reported zero vulnerabilities |
| Dashboard production build | PASS | TypeScript and Vite build completed |
| One-command local startup | PASS | `audioenv app --data-dir ...` applied migrations and served bundled SPA plus API from an isolated wheel install without Node |
| Dashboard product flow | PASS | Playwright created a project, selected resources, preflighted, launched, waited for completion, reloaded the checkpoint, viewed the comparison, and exposed export downloads |
| Persisted local workflow | PASS | 30-step subprocess run completed with 92 persisted metrics, checkpoint/manifest artifacts, two evaluations, and one paired comparison |
| CPU parameter update | PASS | Initial and final parameter hashes differ; checkpoint reload reproduces the trained result |
| Local held-out comparison | PASS, LIMITED | Baseline `0.0`, trained `1.0` over 24 samples from the immutable test split; audio-conditioned platform proof, not a Qwen result |
| Official EnvHarness type integration | PASS | Exact bridge test passed against pinned upstream commit |
| Qwen3-Omni dependency probe | PASS | Dependencies import; probe correctly blocks on missing CUDA and local checkpoint without downloading |
| Real Qwen native-audio episode | BLOCKED | No CUDA device or checkpoint on the audit host |
| VeRL-Omni GSPO update | BLOCKED | Requires separately approved multi-GPU worker |
| Independent caller transfer | NOT RUN | Depends on the real native-audio training run |
| Ten-point native-model learning lift | NOT RUN | Local proof exceeds the threshold but is explicitly ineligible as native Qwen evidence |
| OCI coding sandbox | BLOCKED | No Docker-compatible runtime is installed |

