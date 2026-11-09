# Documentation

## Build, understand and extend

| Document | Purpose |
|---|---|
| [Project README](../README.md) | Native build, measured scope and repository map |
| [Development](development.md) | C++/Python ownership, build modes, tests and contribution rules |
| [Architecture](architecture.md) | Runtime/cache invariants, request lifecycle and telemetry |
| [Controller design](controller-design.md) | Six-action C++ policy, calibration, transitions, probes and API |
| [Script catalog](../scripts/README.md) | Python entry points grouped by responsibility |
| [Upstream audit](upstream.md) | Pinned llama.cpp integration and ownership |

## Run and measure

| Document | Purpose |
|---|---|
| [Benchmarking](benchmarking.md) | Model download, smoke/pilot commands, resume and analysis |
| [Model correctness](model-correctness.md) | Exact token-ID/stop gate and observed coverage |
| [Experiment protocol](experiment.md) | Workloads, timing, baseline selection and scientific gates |
| [Held-out evaluation](evaluation.md) | Full-grid runner, ablations, figures and execution limits |
| [Environments](environments.md) | Codex Cloud development, Codespaces execution and hosted workflows |
| [Codespaces lifecycle](../scripts/codespaces/README.md) | Ownership, credentials, cleanup and shutdown proof |

## Evidence and limitations

| Document | Purpose |
|---|---|
| [Controller results](controller-results.md) | Completed development study, uncertainty, ablations and bottleneck |
| [Evidence index](../results/README.md) | Canonical receipts, immutable raw runs and local examples |
| [Four-page PDF](../results/verification/engineering-report-20261108/tokenomics-engineering-report.pdf) | Reproducible engineering checkpoint |
| [CPU profiles](cpu-profiles.md) | Native dispatch, rejected optimizations and numerical diagnosis |
| [Divergence investigation](divergence-investigation.md) | Earlier verification-batch arithmetic discrepancy |
| [Feasibility decision](feasibility.md) | What completed and partial experiments establish |
| [Requirements](requirements.md) | Original plan mapped to implementation and executed evidence |
| [Technical report](report.md) | Fuller interpretation and implementation history |
| [Related work](related-work.md) | Prior work and overlap |

[Handoff](handoff.md) records current continuation status. [Decisions](decisions.md)
and [engineering history](engineering-history.md) retain older choices and session
snapshots. Historical queue/access states are not instructions to restart runs.
Full held-out, Metal and CUDA model execution remain unperformed.
