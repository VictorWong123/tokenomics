# Frequency-filtered copying: exploratory fit protocol

The existing copy oracle shows opportunities below the conservative five-token
minimum match. This experiment asks whether observed continuation frequency
can filter weak matches without learning from future tokens. It is a chain-only,
request-local hypothesis; it is not a reproduction of the published suffix-tree
systems or their cross-request/tree verification methods.

The grid is frozen in `configs/frequency-copy-fit-v1.json` before execution:
four minimum match lengths, two support counts, three joint confidence ratios
and five caps. Search tries longest suffixes first. A next token must pass both
the support count and the ratio of that entire continuation prefix's occurrences
to the original matched-suffix count. Later missing/censored continuation tokens
cannot improve the ratio. Frequency ties prefer the most recent occurrence.
Only an empty first step permits backing off to a shorter suffix.

Replay uses the same three archived development-fit trajectories per operating
point,64 tokens each, with ordinary/ngram/suffix controls re-audited first.
Proposals read only the current prompt, committed output and pending token.
No single grid cell is selected for deployment. The report retains all cells,
unique cycle/wasted-row Pareto points and aliases; grid cells sharing the same
trajectory are not independent experiments. There is no model inference,
latency estimate, validation result or held-out claim.

The trace retains actual proposed token IDs, effective budgets and accepted
counts. Histograms of verification query-row counts remain available alongside
aggregate work: equal cycle/waste counts can still have different kernel/KV
costs. These histograms are descriptive, without a fitted latency model.

The completed grid contains120 cells per operating point and720 per-prompt
records. Five tests include18360 binary-sequence cases against an independent
prefix-enumeration implementation. The replay reproduces15 measured upstream
aggregate counts per operating point (30 total); the archived minimal logs
contain no detailed cycles to cross-check. All46905 newly computed proposal/
cycle records and query-shape histograms are retained. Matched raw suffix
controls distinguish frequency filtering from merely lowering the match length.

| Operating point / strategy | Cap | Cycles | Wasted verification rows |
|---|---:|---:|---:|
| M3 0.5B, measured ngram control replay | 8 | 133 | 25 |
| M3 0.5B, raw minimum-three suffix | 8 | 133 | 27 |
| M3 0.5B, minimum-three frequency, joint ratio1 | 8 | 133 | 23 |
| M3 0.5B, existing minimum-five suffix | 4 | 142 | 1 |
| Linux 3B, measured ngram control replay | 4 | 142 | 34 |
| Linux 3B, raw minimum-two suffix | 2 | 141 | 17 |
| Linux 3B, minimum-two frequency, joint ratio1/2 | 2 | 141 | 13 |

All rows represent192 canonical output tokens. Frequency filtering removes a
few rows at useful points, but no grid cell improves the existing zero-waste
minimum-five/cap1 point:161 cycles on M3 and169 on Linux. More aggressive copying
can save cycles while wasting many additional rows. Different query shapes can
also have different backend costs at identical aggregate counts. These small
fit-only changes do not establish faster generation or generalization, and no
new runtime drafter is selected. [Full evidence](../results/verification/frequency-copy-fit-20261116/frequency-fit.json).

Reproduce without model inference:

```bash
.venv/bin/python -m analysis.frequency_copy --out results/local/frequency-copy-fit-fresh
.venv/bin/python -m unittest discover -s tests -p test_frequency_copy.py
```
