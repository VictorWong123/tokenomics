# Cheaper ngram development screen — near-neutral shortest cap

[Run 42461982414, attempt 2](https://github.com/VictorWong123/tokenomics/actions/runs/42461982414/attempts/2)
uses source `6a4eea153b001c4897c0346ec09e063b11779e26`, the same 3B target,
driver binary and execution settings as the model-drafting screen, and no draft
model. All 18 exact ID/stop comparisons and 66 original-KV action checks pass.
The ZIP, inner archive and all 134 members have verified hashes/byte counts;
prefixes, continuations and cycle/cache accounting were independently checked.
Hosted finalize separately confirms actual Shutdown. Attempt 1 failed during
runner startup before inference; the unchanged owned lifecycle succeeded on retry.

Geometric mean throughput ratios to the within-run target reference are fixed1
**1.0113**, fixed2 **0.9429**, fixed4 **0.9205**, fixed6 **0.8402**, and fixed8
**0.7821**. Fixed8 is also the existing variable-length lookup comparator.
Cap 1 ratios are QA 1.023, code 1.038, summary 0.974. Its actual proposals are
6/14/18, with 5/14/14 accepted. These are actual trained-model lookups, not
empty-lookup-only execution. Peak screen RSS is 2.18 GiB.

This is three fit prompts, cap 64 and one repetition. The 1.1% aggregate point
estimate is below the frozen 5% practical margin and ambiguous without repeated
validation. Neither Q1 nor Q2 is cleared. The full six-condition Pilot A duration
extrapolation is 12.25 hours. The separate schema-3 snapshot diagnostic reports
insufficient common-prefix checkpoints for a scheduling decision.

`configs/ngram-confirmation-v1.json` freezes fixed1 from this screen, hashes
this fit manifest/analysis, and selects nine different development prompts
(indices 03–05 in each category), both caps 128/512 and three repetitions.
It compares target, frozen fixed1 and fixed8, with balanced order. Its strict
analysis honors fixed1 even if another action wins validation, rejects partial
grids, and cannot clear the full Pilot A/B gate. No held-out prompts are used.
This is a targeted investigation of the ambiguous point, not a replacement
for the full protocol or an adaptive-controller result.

Regenerate the screen analysis:

```bash
.venv/bin/python -m analysis.timing_screen \
  results/verification/codespace-ngram-20261108/qwen25-05-3-q4-ngram/timing \
  --out results/local/observed-ngram-screen
```
