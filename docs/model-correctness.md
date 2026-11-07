# Trained-model differential gate

Run this before interpreting model timings. It compares the ordinary target-only
adapter, with no draft loaded, against resident target-only, fixed caps
`{1,2,4,6,8}`, confidence, acceptance and scripted fallback/re-entry. Both cache
strategies run with identical target weights and execution settings. Each case
runs twice in the same candidate process to test request reset.

```bash
.venv/bin/python scripts/check_models.py \
  --target models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
  --draft models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --threads 2 --out results/local/codespace-model-check-001
```

The default matrix has 272 comparisons: three development prompts (one per
category), four output-budget cases, and one logical-context boundary; 17
configurations, each repeated twice. The boundary uses valid tokenizer IDs
repeated to the requested logical context, not an invented vocabulary mapping.
No held-out outputs are generated. `--max-tokens` defaults to 32 for the three
development prompts. Each subprocess has a 1,800-second timeout; the enclosing
remote workload must also enforce its overall lifecycle budget.

An existing output directory is refused. Use a new attempt directory after a
failure and preserve the previous attempt. Jobs, raw token IDs, cycle records,
stderr, source/binary/model hashes and hardware metadata remain on disk. The
first prompt/token/stop mismatch fails the gate and is recorded; it is never
converted into a successful performance record. Investigate that divergence
before expanding measurements. Script actions are now retained in the engine's
serialized settings as well as the input job file.

Candidate records are checked as soon as the engine flushes each JSONL row.
The first mismatch stops the candidate process; timeouts retain completed
per-request checks and raw partial outputs. A partial check count is not a
passed gate. Failure summaries include the exception type, checked count and
first mismatch, avoiding dependence on archive transfer for basic diagnosis.

`status.json` reports observed zero/partial/full acceptance, catch-up, switching
and stop reasons. A pass does not imply every path occurred on trained weights;
missing coverage needs targeted follow-up. Forced EOG and proposal corruption
remain separate synthetic tests. This gate does not compare the exact stock
executable, establish model quality, answer feasibility, or validate stochastic
sampling. It is a bounded smoke gate before broader differential evaluation.

For local validation of this tooling only, use generated fixture GGUFs and a
small three-category development prompt manifest; pass `--synthetic-fixture`.
Such results are explicitly synthetic, even though they execute real forward
passes. Do not present them as trained-model validation.

The trusted Codespace workflow calls `scripts/model_workload.py`, reusing
the unchanged lifecycle controller's ownership checks and hosted
initialize/finalize jobs. It does not run timing or full pilots. Models persist
under `/workspaces/.tokenomics-model-cache`, outside the clean Actions checkout;
the downloader verifies their hashes on reuse. Available RAM must be at least
5 GiB and free disk at least 4 GiB before the first candidate is downloaded.

Bootstrap has a 30-minute deadline and the model child a 150-minute deadline,
within the 210-minute workload job. Artifacts use unique run/attempt paths and
an exact file allowlist (16 MiB per file, 48 MiB total), retaining partial model
gate outputs on failure. Model bytes, credentials, builds and environments are
excluded. `TOKENOMICS_MODEL_SUMMARY` reports model-stage state; generic finalizer
cleanup reports confirmed shutdown separately, without a fixture-success line.
Neither a successful archive upload nor shutdown proves the model gate passed.

After this gate passes on trained models, measure the initial budget without
starting a full pilot:

```bash
bash run.sh cpu --candidate qwen25-05-15-q4 --threads 2 --timing-only \
  --run-dir results/local/codespace-timing-001
```

Exit 10 means the timing screen completed with resource headroom. It is neither
a feasibility answer nor project completion. The timing batch uses three
development prompts, target-only/fixed4/confidence and cap 64. Inspect its nine
correctness records, duration estimate, actual model lock, RSS and hardware.
`--max-hours` is an estimate screen, not a hard runtime deadline. A remote stage
still needs an outer timeout. Full Pilot A/B should be scheduled only after
inspecting this result and fitting the actual lifecycle budget; do not launch
the 1,080-request pilot blindly on the two-core machine.

On the dedicated Codespace, supply
`--model-dir /workspaces/.tokenomics-model-cache` to reuse the SHA-verified
weights and lock from the model gate. The default remains `models/`. Changing
the result directory never changes the model lock; incompatible results must
not be resumed under different source, binaries, weights or settings.

The trained-model gate has since failed; the fixed workload currently runs the
bounded divergence investigation in `docs/divergence-investigation.md`. Its
`diagnostic_complete` state is evidence collection, never a correctness pass.
The direct 272-comparison gate remains available after the cause is resolved.
