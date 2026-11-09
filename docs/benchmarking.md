# Benchmarking and reproducibility

Prepare the full development environment with `bash scripts/bootstrap.sh cpu`
(or `metal` on macOS). These Python entry points launch the compiled C++
executable; C++ owns each generation request and its online controller.
See [development](development.md) for the runtime-only build and test matrix,
and the [script catalog](../scripts/README.md) for lower-level tools.

## Feasibility entry point

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
# After the controller passes Codex Cloud checks, run its focused trained study:
gh workflow run codespace-experiment.yml --ref main -f phase=controller -f candidate=qwen25-05-3-q4 -f drafter=ngram
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
evaluation. The user authorized an experimental C++ controller despite the
negative feasibility evidence; its [design, implementation and calibration contract](controller-design.md)
are documented. Full feasibility and held-out conclusions remain pending.

The workflow deliberately stops at the scientific decision. Exit 20 means Q1
and Q2 support proceeding with controller implementation; 21 means speculation
helps but cycle-level prediction is unresolved; 22 means review the bounded
investigation before a scope decision. Exit 2 means an execution/access blocker,
12 a user-specified duration limit, 130 an interruption. Exit 0 is reserved for
successful fixture checks; dedicated confirmation/controller scripts also return
0 for their own bounded completed grids. It **does not mean the full project is
complete**. The feasibility script and separate experimental controller study
report their own evidence; held-out evaluation is still separate.

Manual smoke after model download:

```bash
.venv/bin/python scripts/download_models.py --model qwen25-05
.venv/bin/python scripts/download_models.py --model qwen25-15
build-cpu/tokenomics --target models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
  --draft models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --policy fixed --k 4 --max-tokens 128 --prompt 'Explain binary search.'
```

Before interpreting trained-model timings, run the [differential model gate](model-correctness.md).
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
also retains unfavorable comparisons. See [the protocol](experiment.md)
for uncertainty, invalid runs, prompt provenance, baseline deviations and gates.

Copy `models/models.lock.json` and the exact project commit between machines;
the downloader then fetches the same pinned GGUFs. Transfer completed result
directories as archives (`tar -czf linux-pilot.tgz results/local/linux-pilot`),
outside Git. Each includes model/binary hashes, hardware and frozen settings.
Do not combine Linux and Metal measurements into one performance estimate.

