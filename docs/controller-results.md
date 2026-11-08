# Controller engineering result

The experimental C++ controller is implemented and verified on trained models.
On the completed six-prompt development comparison, it exceeds cap8 ngram
throughput by 20.5%, stays within the declared practical margin of the
frozen fixed1 baseline, and runs 2.9% below ordinary target-only decoding.
This operating point does not establish an acceleration over ordinary decoding.

## Scope and provenance

- Source: `e98c0fa84a2c2ff52df198f19eae6ce72d51c83c`.
- Run 42492089188, attempts 1/2. Attempt1 preserves 25 complete blocks; attempt2
  finishes the same frozen grid. All 25 earlier block receipts/raw files, the
  prior and fit measurements are byte-identical after resume.
- CPU: the dedicated Linux Codespace, AMD EPYC 7763, two logical CPUs on one
  physical core, two inference threads, 7.8 GiB RAM, no swap/GPU.
- Target: Qwen2.5-3B-Instruct Q4_K_M, ngram-simple drafting, retained-cache
  operating point, context 2048, batch 512, greedy/F16 KV.
- Fit: three designated development prompts, cap 64; original-state action
  replays freeze all six priors and select fixed1 from full-request measurements.
- Validation: six distinct development prompts, two per category, caps 64/128,
  three repeats, seven conditions. All 252 records pass exact IDs/stops.
- Separate logging grid: two declared median/worst prompts, cap 128, two repeats,
  four conditions, 16 records. These are fresh diagnostic histories.

The [design/API](controller-design.md) explains action scoring, full one-cycle
transition charging, EWMA 0.2, four-cycle holds, 5% hysteresis, rotating probes,
reset, eligibility, counters and limitations. Both trained mechanisms also pass
28 focused correctness records each, including all six actions; the model path
synchronizes 40 draft tokens. Those gates use explicitly synthetic priors to
exercise behavior, and their timings are excluded from the performance table.

## Complete-grid measurements

The endpoint is the geometric mean of prompt/cap median decode-throughput
ratios. The 95% intervals resample distinct prompts with both caps together.
Repetitions are not independent prompts. All inference, probes, synchronization,
updates and reconciliation remain charged; no slow records were removed.

| Cost-aware compared with | Throughput ratio | Prompt-bootstrap 95% interval |
|---|---:|---:|
| Ordinary target-only | 0.9709 | [0.9607,0.9810] |
| Development-frozen fixed1 | 0.9943 | [0.9853,1.0023] |
| Existing variable-length ngram, cap8 | 1.2048 | [1.1271,1.2868] |
| Acceptance heuristic | 1.0272 | [1.0157,1.0403] |

All twelve cost-aware/fixed1 cells are within the declared 5% practical margin.
Against ordinary decoding, ten cells are ties and two losses; no cell is more
than 10% slower by time/output token. The worst throughput ratio is 0.9360.
The acceptance comparison is statistically positive but below the proposed 5%
practical aggregate criterion. Beating cap8 does not imply beating the tuned
fixed baseline or ordinary decoding.

| Condition | Decode tokens/s¹ | Median warm completion ms | Peak lifetime RSS GiB |
|---|---:|---:|---:|
| Ordinary target | 7.673 | 19981.8 | 2.17 |
| Fixed1 | 7.492 | 20208.8 | 2.17 |
| Ngram cap8 | 6.183 | 23849.3 | 2.18 |
| Acceptance | 7.252 | 20838.8 | 2.17 |
| Cost-aware | 7.450 | 20432.6 | 2.18 |
| No target-only fallback | 7.427 | 21431.3 | 2.17 |
| No probes | 7.477 | 20263.3 | 2.17 |

¹ Geometric mean of cell median throughput. Warm latency includes prompt
processing, controller setup and generation. RSS is a process-lifetime peak
including load/warm-up, not policy-only memory or GPU VRAM.

No-fallback/cost-aware is 0.9970 [0.9933,1.0005]; no-probes/cost-aware is 1.0036
[0.9953,1.0118]. These ablations do not show a practical aggregate improvement.
Detailed/minimal throughput is 1.0015 [0.9868,1.0165] on the separate two-prompt
logging grid; a small negative estimated logging cost is timing noise.

## What the implementation actually does

Across the 36 measured cost-aware requests, it emits 3456 exact tokens in
3252 cycles. Ordinary fallback supplies 1634 tokens (47.3%); mandatory tails and
probe action0 are counted separately. It performs 175 probes and 99 normal
incumbent switches. Detailed traces expose eligibility, candidate scores,
reasons, prediction, full outcomes and chosen-action-only updates.

Decision plus update work averages **1.67 microseconds per cycle**, totaling
5.43 ms across 476.19 seconds of generation (0.00114%). Mean policy initialization
is 0.83 microseconds and configuration parsing 8.63 microseconds per request,
inside warm latency. These are measured software clocks for this point, not a
cross-machine guarantee. The policy's fixed-array select/observe path does not
allocate, while initialization/JSON/runtime processing may allocate.

The controller's mean absolute fractional cycle-time prediction error is 22.7%.
It is an experimental online estimator, not an optimal scheduler or a proven
predictor of generation phases. With six eligible actions and an unchanged
incumbent, a particular alternative is normally revisited every 80 cycles under
the default rotating schedule. A short request may provide only one observation
of a longer action; fast learned recovery is not assumed.

Probe cycles consume 32.79 seconds, 6.89% of total generation. This is their full
charged execution time, **not incremental probe overhead**: they also commit
necessary output. The matched no-probe comparison is the appropriate overhead
ablation. Ngram has no separate draft-model catch-up cost; the model correctness
gate establishes that path without claiming model-drafter performance.

## Why further wrapper tuning is insufficient

| Condition | Output tokens | Target rows evaluated | Unaccepted proposals |
|---|---:|---:|---:|
| Ordinary target | 3456 | 3456 | 0 |
| Fixed1 | 3456 | 3576 | 120 |
| Ngram cap8 | 3456 | 4572 | 1116 |
| Acceptance | 3456 | 3723 | 267 |
| Cost-aware | 3456 | 3604 | 148 |

Cost-aware evaluates 21.2% fewer target rows than cap8 and avoids most of its
unaccepted proposals. It still evaluates 4.3% more rows than ordinary decoding.
Verification occupies 99.992% of its measured generation time; lookup and policy
bookkeeping are small. The guarded CPU's limited verification batching savings
cannot compensate for rejected rows. High acceptance alone is not the objective.

The next experiment targets native target execution: a separately gated,
row-consistent packed CPU profile. Its ordinary target baseline must receive
that same optimization. A paired study compares both profiles, exact output
IDs, warm latency, decode rate, logging and memory. It cannot manufacture a
policy gain by retaining a slower ordinary baseline. See [CPU profiles](cpu-profiles.md).

## Reproduce and inspect

The [canonical receipt](../results/verification/codespace-controller-20261108/verification.json)
records artifact ZIP/inner archive hashes, source/model/binary identity,
regenerated statistics and actual Shutdown from finalizer job 128815104360.
All 1587 archive members and raw block/IDs/stops/cache checks pass independent
verification. The hosted job's success alone was not used as shutdown proof.
Read-only [hosted re-verification](https://github.com/VictorWong123/tokenomics/actions/runs/42509187514)
also passes against the retained evidence, independently reconstructing the prior
and summary. It starts no inference or Codespace.

- [Raw attempt2](../results/verification/codespace-controller-20261108/attempt-2).
- [Audited primary table and figures](../results/verification/codespace-controller-20261108/derived).
- [Category, memory, acceptance/cost and regression figures](../results/verification/codespace-controller-20261108/derived/details).
- [Byte-stable resume check](../results/verification/codespace-controller-20261108/derived/resume-verification.json).

Without models or inference:

```bash
.venv/bin/python scripts/demo_controller.py --max-cycles 12
.venv/bin/python -m analysis.controller \
  results/verification/codespace-controller-20261108/attempt-2/validation \
  --out results/local/reproduced-controller
.venv/bin/python -m analysis.controller_details \
  results/verification/codespace-controller-20261108/attempt-2/validation \
  --fit results/verification/codespace-controller-20261108/attempt-2/fit \
  --out results/local/reproduced-controller-details
```

The full 90-prompt held-out matrix, full Pilot A/B decision and Mac/Metal
measurements remain unexecuted. These are bounded development engineering
results. No universal acceleration, model-quality, energy, dollar-cost or research
novelty claim follows. The controller remains a reusable C++ implementation with
explained behavior and a measured operating-point limitation.
