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

## Q4 pair research experiment

`q4_pair_study.py` builds the opt-in guarded Q4_K paired verification path,
runs the existing 272-case trained gate, and collects the fixed 72-record
development comparison in `configs/q4-pair-screen-v1.json`. Use the explicit
`q4-pair` phase of the owned experiment workflow on the research branch. It
does not change the default profile or establish a held-out/controller gain.

## Local M3 Metal research experiment

`metal_short_mv_study.py --model PATH --out FRESH_DIRECTORY` builds the guarded
and opt-in short-MV profiles on native Apple Silicon, runs correctness checks,
and enables the 72-record development screen only after cross-profile equality.
Use `--gate-only` to omit timings. `check_metal_profile.py` runs that gate against
already-built binaries. See [M3 instructions and limitations](../docs/metal-short-mv.md).
For the 16 November research session, the user's no-Codespaces instruction
supersedes the older workflow route above.


## Conservative suffix proposal experiment

`suffix_study.py --model PATH --out FRESH_DIRECTORY` checks the opt-in suffix
proposal source on the default guarded M3 backend, then compares it with
ordinary decoding and unchanged ngram cap1/cap8. `--gate-only` omits timings.
It uses `build-metal/tokenomics` unless `--binary` is supplied. See
[the suffix design, fit evidence and commands](../docs/suffix-drafter.md).
`python -m analysis.proposal_replay --out FRESH_DIRECTORY` reproduces the
archived development-fit proposal comparison without loading models.

## Target greedy sampler experiment

The opt-in `TOKENOMICS_GREEDY_NO_TEMP` profile removes the redundant target
zero-temperature filter while preserving upstream greedy verification.
`check_metal_profile.py --candidate-profile greedy` gates it against the saved
guarded reference. `test_greedy_equivalence` checks the upstream selection rule;
its optional `--bench` mode measures synthetic host candidate/sampler cost.
See [profile scope and local commands](../docs/greedy-sampler.md).

`check_suffix_replay.py` executes the production C++ suffix function on archived
canonical validation IDs, using the optional `test_suffix_replay` target. It
loads no model and checks the native counts against the frozen Python replay.
