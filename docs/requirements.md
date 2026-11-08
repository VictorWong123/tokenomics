# Project requirements and evidence

This maps the attached project plan to the implemented system. The user authorized experimental controller completion despite the negative
feasibility evidence. Supporting tooling alone does not complete a research stage.
See [the technical report](report.md) for interpretation and
[feasibility](feasibility.md) for the current decision.

| Requirement | Status and evidence |
|---|---|
| Pin execution code, model provenance and settings | Implemented: clean llama.cpp pin, model catalog/locks, executable/source/model hashes and frozen manifests |
| Portable C++20 inference-loop integration | Implemented and tested on Linux CPU; Metal/CUDA build options exist, execution remains unverified |
| Upstream greedy verification | Implemented: the adapter calls upstream sample-and-accept; no second verifier |
| Fixed caps and existing dynamic/acceptance baselines | Implemented; actual proposal counts and stopping are logged |
| Target-only fallback, retained KV, lazy catch-up, re-entry and reset | Implemented; synthetic edge cases and trained differential checks cover the executed paths |
| EOS, budgets, context boundary, rejection and cache accounting | 906 fixture differential cases plus focused comparisons; separate trained gate has 272 cases. Unobserved real-model coverage remains identified in each gate report |
| Original-history common-prefix replay | Implemented: bounded target/draft snapshots; trained 3B screens validate 66 action continuations per point |
| Candidate benefit and scheduling opportunity | Incomplete: independent-model screens lose; the completed nine-prompt frozen ngram confirmation fails the practical-gain criterion; full Pilot A/B and Q2 remain unresolved |
| Cost-aware EWMA controller, priors, probes and hysteresis | Implemented experimentally by explicit user override: pure C++ six-action EWMA, full transition charge, hold/hysteresis, rotating probes, resets, explicit frozen priors and detailed/minimal telemetry; see controller design |
| Online transition scoring and controller ablations | Implemented: no-fallback/no-probe variants, actual sync-separated updates and operating-point binding. Full trained comparison is pending |
| Frozen 40-development / 90-held-out workload | Implemented with original CC0 prompts and embedded summary sources; held-out measurements have not run |
| Balanced ordering, resumability and complete-grid analysis | Implemented; full/partial fixture runs, immutable blocks and real resumed confirmation are checked |
| Fixed baseline selected on development only | Implemented; validation/held-out paths reject retuning and incomplete grids |
| Timing, RSS, minimal/detailed logging | Implemented with documented boundaries. Paired trained logging/profile measurement is prepared, not yet complete |
| Stock executable | Optional unmodified diagnostic builds and stopping-semantic investigation exist; no equivalent-endpoint stock performance claim |
| CPU performance optimization | Buffer reuse is implemented. Row-consistent packed verification is experimental; trained run was cancelled and shutdown confirmed; revalidation is deferred until the controller passes Cloud checks. Rejected alternatives and raw diagnostics are retained |
| Final held-out result, controller figures and resume claim | Bounded controller study and figures implemented; trained execution and full held-out evaluation remain incomplete. Current plots/demo show only their labeled executed experiments; no cost-aware speedup, research novelty or M3 execution is claimed |
| Reproducible documentation and evidence | Build/CLI/runtime/experiment/lifecycle docs, raw artifacts, verification receipts and decision records are maintained alongside code |

## Reading guide

- [Build and run](../README.md): bootstrap, real models and demonstrations.
- [Workspace and test environment](environments.md): Codex Cloud, Codespaces,
  workflows, credentials, artifacts and recovery.
- [Runtime contract](architecture.md): pending token, caches, policies,
  snapshots, timer boundaries, JSON output and RSS.
- [Controller design and use](controller-design.md): policy contract, calibration,
  provenance, implementation, CLI and interpretation.
- [CPU profiles](cpu-profiles.md): dispatch changes, exactness gates,
  comparison protocol and rejected optimizations.
- [Model correctness](model-correctness.md): trained cases and their limits.
- [Frozen experiment protocol](experiment.md): workload, baselines, pilots,
  statistics and invalid-run rules.
- [Related work](related-work.md): primary sources and overlap.
- [Report](report.md), [decision](feasibility.md) and [handoff](handoff.md):
  measured outcomes and the work still required.
