# AudioEnv V1 milestone plan

| Milestone | Deliverable | Completion evidence |
| --- | --- | --- |
| 0 | Audit, architecture, plan, acceptance criteria | Four reviewed documents plus V0 test/build output |
| 1 | Persistence, lifecycle, artifacts, API, worker, dashboard shell | API/browser tests and one-command local startup |
| 2 | V1 IR, custom authoring, validation, τ² adapter, EnvHarness proof | Oracle/adversarial validation and immutable split manifest |
| 3 | Qwen3 native caller/agent execution | GPU trace containing real bidirectional speech and a tool call |
| 4 | CPU trainer and VeRL-Omni GSPO adapter | Parameter delta, reload parity, then GPU LoRA update evidence |
| 5 | Held-out evaluation, comparisons, trace UI, exports | Persisted evidence behind every displayed metric |
| 6 | Frozen end-to-end release workflow | Reproducibility bundle and completed acceptance report |

Implementation proceeds vertically: a local environment is created, trained,
evaluated, persisted, served, and displayed before additional integrations are
added. Milestones 3, the production half of 4, and 6 cannot be marked complete
on the current non-CUDA host.

Pinned research integration revisions at audit time:

- EnvHarness `fab7d57441f06b75c73a900e04561d4d7600f361`.
- VeRL-Omni `f31a6116119fd62a89554987f51ae279a567d66b`.
- Qwen checkpoint revisions must be captured by the model probe before use.
- τ² source and task hashes must be captured during import rather than relying
  on an unpinned `main` checkout.

## Current milestone status

- Milestone 0: complete and verified.
- Milestone 1: local foundation complete; multi-process recovery is covered by
  persisted heartbeat logic and tests, while cloud execution is intentionally
  absent.
- Milestone 2: custom authoring and EnvHarness proof complete. The pinned τ²
  source importer is complete; its isolated executable tool/verifier worker is
  still open.
- Milestone 3: adapter and preflight code complete, GPU execution blocked.
- Milestone 4: CPU trainer complete, VeRL-Omni launch boundary complete, GPU
  optimizer execution blocked.
- Milestone 5: local metrics, evaluations, artifacts, comparisons, and dashboard
  complete for the CPU slice; independent native-audio evaluation remains open.
- Milestone 6: local workflow complete; native Qwen release workflow blocked on
  approved CUDA infrastructure.

