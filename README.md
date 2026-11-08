# Tokenomics

Portable C++20 speculative-decoding runtime engineering on a pinned llama.cpp.
It keeps greedy verification upstream and implements draft-cache retention,
checked rollback, lazy synchronization, target-only/re-entry, and policy hooks.
Diagnostic action replay restores original target KV checkpoints and a shared
prepared draft state instead of repeatedly reconstructing the combined prefix.
CPU is the initial reference. macOS Metal is supported by configuration;
CUDA is optional. Linux cloud, Codespaces and Mac results are separate.

**Status: in progress.** Cloud Release and sanitizer checks pass, including
468 differential fixture cases. Codespaces passed the 272-comparison trained
correctness gate and trained original-KV replay checks. A completed nine-prompt
confirmation passes all 162 exact comparisons, but its frozen cap1 throughput
ratio is 0.9846 [0.9721, 1.0019] to target-only and fails the 5% practical-gain
criterion. The opt-in packed CPU profile is undergoing separate trained checks.
Full Pilot A/B, cost-aware scheduling, held-out evaluation and Metal execution
remain incomplete. See [the handoff](docs/handoff.md),
[feasibility](docs/feasibility.md) and the
[verified raw confirmation](results/verification/codespace-confirmation-20261108).

The [requirements and documentation index](docs/requirements.md) distinguishes
implemented behavior, measured evidence and remaining research work. See
[workspace/test-environment roles](docs/environments.md) and
[CPU profiles](docs/cpu-profiles.md) for operational details.

## Build and check

Linux needs Git, a C++20 compiler, Python 3.12+, a Python venv implementation and
curl. macOS needs Xcode Command Line Tools (`xcode-select --install`) and Python
3.12+ (for example `brew install python@3.12`). No NVIDIA GPU is required.

```bash
git clone --recurse-submodules --branch main https://github.com/VictorWong123/tokenomics.git
cd tokenomics
bash scripts/bootstrap.sh cpu              # Linux
# bash scripts/bootstrap.sh metal          # macOS, Metal enabled
```

The bootstrap creates `.venv`, installs pinned Python packages and CMake/Ninja,
builds `build-cpu` or `build-metal`, and runs unit, differential runtime and
tooling tests. The default is four build/runtime threads. Set
`TOKENOMICS_BUILD_JOBS=2` if compilation memory is limited. The devcontainer
includes a compiler and prerequisites, then calls the same bootstrap.

No-model end-to-end smoke (generated fixtures are not language models):

```bash
bash run.sh cpu --fixtures-only --run-dir results/local/linux-fixtures
# bash run.sh metal --fixtures-only --run-dir results/local/mac-fixtures
```

Terminal state-transition demonstration:

```bash
.venv/bin/python scripts/demo.py --build-dir build-cpu
# Verify and replay the retained trained-model trace, with no inference/download:
.venv/bin/python scripts/demo.py --recorded
```

The recorded demo shows scripted fallback, partial/full acceptance and lazy
catch-up from the passing trained gate. It verifies original artifact file
hashes and the selected request against ordinary target IDs before display.
It demonstrates correctness mechanics; the cost-aware policy remains gated.

## Real-model feasibility: one entry point

```bash
bash run.sh cpu --run-dir results/local/linux-pilot
# On the Mac:
# bash run.sh metal --run-dir results/local/mac-metal-pilot
```

The script checks hardware/RAM, verifies models against Hub LFS SHA-256,
checks frozen prompt/context fit, times a small batch and estimates duration,
then runs both feasibility investigations. It uses at most two memory-safe
independent-model candidates; if both are unprofitable, it investigates upstream
ngram lookup drafting. `--candidate qwen25-05-15-q4` restricts the investigation;
`--threads 2` changes the frozen runtime allocation. `--max-hours 8` stops after
the timing batch if the rough Pilot A estimate exceeds that budget.

**Rerun the identical command to resume.** Each request is saved and each
complete comparison block is committed atomically. An incomplete/failed block
gets a new retained attempt; slow runs are not excluded. Code, binary, model, prompt
or configuration changes require a new run directory. Raw local results and
models are ignored by Git.

`--budget-seconds 6600` pauses a long pilot at complete comparison-block or
checkpoint-group boundaries (exit 11). An unchanged rerun resumes saved work;
the deadline is an execution allowance, not a change to the frozen experiment.
It reserves the largest observed group duration before starting another group;
an unusually slow first/new group can still reach the outer execution timeout.

Develop and run portable checks in Codex Cloud, then push checked stages to
`main`. Use the existing Codespace only to execute trained-model tests and
measurements through its owned Actions lifecycle:

```bash
gh workflow run codespace-experiment.yml --ref main -f phase=timing
# Screen the larger memory-safe candidate and validate checkpoint replay:
gh workflow run codespace-experiment.yml --ref main -f phase=timing -f candidate=qwen25-05-3-q4
gh workflow run codespace-experiment.yml --ref main -f phase=timing -f candidate=qwen25-05-3-q4 -f drafter=ngram
gh workflow run codespace-experiment.yml --ref main -f phase=confirmation -f candidate=qwen25-05-3-q4 -f drafter=ngram
gh workflow run codespace-experiment.yml --ref main -f phase=pilot
```

The fixed pilot investigates the versioned candidates and cheaper ngram mode.
`candidate=all` is the pilot default; either catalog candidate can be selected
explicitly. Timing screens also validate original-state replay on trained
weights when enough continuation is available; this is not a Q2 feasibility
decision from three prompts.
Each invocation retains partial evidence and shuts down the same dedicated
Codespace. Pilot data persists under `/workspaces/.tokenomics-experiment-cache/`
by source SHA; rerun that exact SHA to resume. Code changes require a fresh run.
Artifacts are retained for 30 days. This is development data, not held-out
evaluation. Full controller implementation remains conditional on both gates.

The workflow deliberately stops at the scientific decision. Exit 20 means Q1
and Q2 support proceeding with controller implementation; 21 means speculation
helps but cycle-level prediction is unresolved; 22 means review the bounded
investigation before a scope decision. Exit 2 means an execution/access blocker,
12 a user-specified duration limit, 130 an interruption. Exit 0 is reserved for
the fixture workflow and **does not mean the full project is complete**.
This script runs the feasibility phase; controller implementation and held-out
evaluation follow its evidence, not an automatic success message.

Manual smoke after model download:

```bash
.venv/bin/python scripts/download_models.py --model qwen25-05
.venv/bin/python scripts/download_models.py --model qwen25-15
build-cpu/tokenomics --target models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
  --draft models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --policy fixed --k 4 --max-tokens 128 --prompt 'Explain binary search.'
```

Before interpreting trained-model timings, run the [differential model gate](docs/model-correctness.md).
It checks exact IDs, stops, repeated-request resets and cycle/cache accounting
across 272 comparisons. Its observed coverage report distinguishes exercised
paths from unverified ones. `run.sh ... --timing-only` then stops after the small
development timing screen (exit 10), without launching full Pilot A/B.

Show all C++ options with `build-cpu/tokenomics --help`. Jobs in JSONL can
change policy/cap per request while retaining model weights. The Python runner
uses separate processes for fair target-alone versus resident-draft memory
comparisons, excluding a warm-up and retaining cold-load and whole-process costs.

## Analysis and transfer

```bash
.venv/bin/python -m analysis.report results/local/linux-pilot/qwen25-05-15-q4/pilot-a
# A checked-in example is available without model downloads:
.venv/bin/python -m analysis.report results/example-fixtures --out results/local/example-analysis
# Observed trained-model timing screen (preliminary, unfavorable):
.venv/bin/python -m analysis.timing_screen \
  results/verification/codespace-timing-20261108/run \
  --out results/local/observed-timing-screen
```

The example contains observed synthetic-fixture timings for analysis testing;
it is explicitly unsuitable for feasibility/speedup claims. Plot generation
also retains unfavorable comparisons. See [the protocol](docs/experiment.md)
for uncertainty, invalid runs, prompt provenance, baseline deviations and gates.

Copy `models/models.lock.json` and the exact project commit between machines;
the downloader then fetches the same pinned GGUFs. Transfer completed result
directories as archives (`tar -czf linux-pilot.tgz results/local/linux-pilot`),
outside Git. Each includes model/binary hashes, hardware and frozen settings.
Do not combine Linux and Metal measurements into one performance estimate.

## Ownership and limits

Our source is in `src/`; upstream inference, drafting and verification are in
the unmodified submodule. [The source audit](docs/upstream.md) explains exactly
what is reused. The `cache=upstream` control reproduces the supported example's
cache behavior; it is not an execution of the unmodified stock executable.
Full-attention transformers with arbitrary partial KV removal are supported;
hybrid/recurrent models, stochastic sampling and context shifting are rejected.

JSON `token_ids` are authoritative. If a budget stops inside a byte-encoded
UTF-8 character, display `text` uses a replacement character and
`text_utf8_repaired` is true; the token sequence is preserved exactly.

The [architecture](docs/architecture.md) explains cache invariants and timing.
All original prompt text is CC0, including original summary documents;
the software is MIT, llama.cpp retains its own license, and model licenses
remain upstream. The 3B Qwen candidate has different terms from smaller Qwen
models; inspect its license before use or redistribution. No weights are in Git.

Development formatting uses `requirements-dev.txt`, Black and clang-format on
owned code. The container base tag/system APT packages are versioned inputs but
not an immutable full OS lock; run manifests retain the actual compiler/build.
