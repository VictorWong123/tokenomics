# Script catalog

These tools support the C++ executable. Model computation, greedy verification,
KV mutation and online scheduling execute natively. Python organizes experiments
and handles files/statistics outside that loop. Paths remain stable for workflows,
tests and frozen evidence; common helpers live in `bench/` and `analysis/`.

## Normal entry points

| Tool | Responsibility |
|---|---|
| `bootstrap.sh` | Shell setup, pinned development packages, CMake build and portable checks |
| `download_models.py` | Download catalog models; verify SHA-256 and retain a model lock |
| `run_all.py` | Fixture workflow or bounded model feasibility pipeline; `../run.sh` bootstraps and invokes it |
| `controller_study.py` | Focused trained correctness, separate fit/validation calibration and controller measurements |
| `evaluate_controller.py` | Frozen full held-out matrix from compatible development priors |
| `demo.py` | Scripted synthetic runtime demo or verified recorded trained trace |
| `demo_controller.py` | Display audited recorded controller decisions without model inference |

Start with [benchmarking](../docs/benchmarking.md),
[controller design](../docs/controller-design.md) or
[held-out evaluation](../docs/evaluation.md). Most CLI tools expose `--help`.
The fixed lifecycle/workload tools below take their identity from workflow inputs.

## Correctness and diagnostic experiments

| Tool | Responsibility |
|---|---|
| `check_models.py` | Launch the native differential model gate; compare exact IDs/stops and cache/cycle accounting |
| `confirm_ngram.py` | Development-frozen fixed-action confirmation; refuses retuning/partial summaries |
| `diagnose_models.py` | Original-history logits and action replay around a model discrepancy |
| `check_kernel_profile.py` | Experimental packed CPU trained gate and preliminary screen |
| `compare_cpu_profiles.py` | Frozen guarded/packed component comparison; requires exact binary/model provenance |
| `profile_diagnosis.py` | Reproduce both original histories from the retained failed paired comparison |

These are specialized, versioned investigations. They are not alternate model
implementations. The packed comparison failed cross-profile equality; see
[CPU profiles](../docs/cpu-profiles.md) before starting a new point.

## Evidence, examples and repository checks

| Tool | Responsibility |
|---|---|
| `verify_controller_evidence.py` | Audit retained controller grids/calibration and regenerate their statistics |
| `verify_profile_evidence.py` | Audit paired-profile successes/failures and original-history diagnosis |
| `verify_experiment_evidence.py` | Check allowlisted GitHub archives, immutable payloads, identities and shutdown proof; starts no inference |
| `make_example.py` | Curate actual synthetic-fixture observations for the no-download analysis example |
| `check_staged.py` | Reject staged models, builds, local runs, environments, oversized files and private keys |

Artifact auditing checks saved evidence. Upstream greedy token acceptance is
still performed by C++ during generation.

## Codespaces operations

`model_workload.py` and `experiment_workload.py` are fixed workflow workloads.
`codespaces/controller.py` owns runner/start/stop lifecycle; despite its name,
it is unrelated to the decoding policy in `src/policy.cpp`.
`codespaces/report_failure.py` classifies operational failures, and
`codespaces/test_controller.py` tests the lifecycle contract.

Use the [owned lifecycle](codespaces/README.md) and
[environment guide](../docs/environments.md). Do not invoke fixed workloads as
ordinary benchmark CLIs or copy lifecycle credentials into the development shell.
The workflow YAML is orchestration; the self-hosted executable performs inference.
