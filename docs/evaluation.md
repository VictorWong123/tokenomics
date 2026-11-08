# Frozen controller evaluation and analysis

The bounded trained development study and the full held-out experiment have
separate completion criteria. The former supplies a first engineering result;
it does not become a 90-prompt result by changing its label.

| Protocol | Prompts | Output caps | Repetitions | Ngram conditions | Measured records |
|---|---:|---|---:|---:|---:|
| Bounded development validation | 6, separate from 3 calibration prompts | 64,128 | 3 | 7 | 252 |
| Full held-out evaluation | 90, 30 per category | 128,512 | at least 5 | 7 | 6300 |

The full model-drafter matrix additionally includes the resident target-only
control and upstream-style cache-replay control. The latter reproduces that
cache strategy in this adapter; it is not a claim of running the stock executable
with equivalent stopping/counting. The stock example remains a separate
diagnostic with its endpoint limitation documented in the report.

## Freeze and execute

Finish compatible development calibration first. Export its prior with
`python -m bench.controller` as documented in the [controller design](controller-design.md).
The prior must be non-synthetic for real performance, and must match the exact
executable, weights, backend, context/batch/thread settings and retained cache
mode. Changing these requires a new calibration, not editing its hashes.

The evaluation entry point freezes the configuration before reading any result.
It uses all 90 versioned held-out prompts, caps128/512 and at least five repeats.
It obtains execution settings and baseline selection from the development prior;
it cannot select a better fixed action from held-out outcomes. Every independent
request resets policy learning and KV state. Baselines, acceptance, cost-aware,
no-fallback and no-probe conditions share the same target and generation boundary.

```bash
.venv/bin/python scripts/evaluate_controller.py \
  --exe build-cpu/tokenomics --target /path/to/target.gguf \
  --prior /path/to/development/prior.json \
  --out results/local/frozen-evaluation --budget-seconds 6600
```

For independent-model drafting add `--draft /path/to/draft.gguf` and use its own
calibration. The script currently exposes local execution; it does not start a
Codespace or create a new machine. Use the owned lifecycle for agent-controlled
Codespaces execution. Full held-out execution has not been performed merely
because this command is implemented and fixture-tested.

A planned deadline pauses at a complete comparison-block boundary (exit11).
Rerun the identical command and inputs to resume. Complete blocks and completion
receipts remain immutable. A failed block retains its attempt; a changed binary,
model, source, prompt or configuration refuses resume. Exact greedy IDs, prompt
IDs and stops must match the ordinary reference throughout the complete matrix
before any summary is produced. Slow measurements remain included.

The earlier guarded ngram confirmation averaged roughly 62–76 seconds of whole
process time per condition, including loading, excluded warm-up and generation.
A rough planning extrapolation to seven conditions ×90 prompts ×2 caps ×5
repeats is about 119 hours at that older point. It is not a controller timing or
an estimate guaranteed for a new kernel/model/workload. The first controller
study is therefore explicitly bounded; its result and bottleneck analysis should
inform whether this large evaluation is justified. No full held-out claim is
made from a partial or shortened grid.

## Regenerate analysis without inference

```bash
.venv/bin/python -m analysis.controller results/local/frozen-evaluation/evaluation \
  --out results/local/evaluation-analysis
.venv/bin/python -m analysis.controller_details \
  results/local/frozen-evaluation/evaluation \
  --fit /path/to/development/fit --out results/local/evaluation-details
```

The same analysis accepts a complete bounded study's `validation` directory and
labels it development. It audits immutable block hashes and raw request records,
checks exact IDs/stops/budgets/cache totals, and enforces calibration identity.
`--no-plots` on the first command generates audited statistics without importing
Matplotlib. Raw measurements are not rewritten by these commands.

The primary report retains paired geometric-mean throughput ratios and
prompt-bootstrap intervals, warm latency, lifetime peak RSS, natural output
counts, action usage, charged stage costs and both ablations. Supplemental
figures add per-category regression distributions, fixed-cap fit cost/acceptance,
memory, and ablation intervals. Their JSON includes ms/output token, first-commit
latency and measured decision/update nanoseconds per completed cycle. Repeats
are collapsed before resampling distinct prompts; both caps stay together.

Fixed-cap curves use development fit data and remain labeled as such. Median and
worst-regression trace selection follows the declared rule, with detailed fresh
requests identified separately from original minimal timing histories. No quality,
energy, dollar-cost, full-precision equivalence or cross-hardware improvement
follows from a throughput ratio. The retained raw records and analysis explain
where execution time and regressions actually arise.
