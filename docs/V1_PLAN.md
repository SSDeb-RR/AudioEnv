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
- VeRL-Omni `eebe8a983b742960b0887026191517fe66505260`.
- Qwen checkpoint revisions must be captured by the model probe before use.
- τ² source and task hashes must be captured during import rather than relying
  on an unpinned `main` checkout.

## Current milestone status

- Milestone 0: complete and verified.
- Milestone 1: complete for the local product. `audioenv app` packages and
  serves the SPA/API, migrations, lifecycle, logs, projects, and recovery.
- Milestone 2: reference V2, custom authoring, and EnvHarness proof complete.
  The pinned τ² Python 3.12 JSONL worker is implemented but not executed on the
  current host, so upstream parity remains partial.
- Milestone 3: adapter and preflight code complete, GPU execution blocked.
- Milestone 4: non-oracle CPU trainer, resume/checkpoint state, and reload gate
  complete. VeRL agent-loop batch bridge is implemented; GPU execution blocked.
- Milestone 5: local logs, metrics, 24-sample held-out evaluations, compatible
  paired comparison, artifacts, and dashboard workflow complete.
- Milestone 6: local workflow complete; native Qwen release workflow blocked on
  approved CUDA infrastructure.

