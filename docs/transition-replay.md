# Exploratory linear transition composition

This bounded follow-up asks whether a single-chain verifier can use a small
request-local transition graph to compose known edges into a path absent from
any single observed span. The idea is related to the token adjacency structure
in [Token Recycling](https://aclanthology.org/2025.acl-long.338.pdf), but its
published BFS draft trees and branch verification are not implemented here.
No published speedup or algorithm reproduction is claimed.

The graph records the most recent observed successor for contexts of one or
two tokens. It initially contains only prompt transitions and then learns only
committed canonical output transitions. Draft construction walks a frozen graph;
its own proposals never train the table. All cycles, including repeated-token
loops, are bounded by the same cap and correction reservation as other drafters.
This differs from suffix copying, which requires an entire contiguous source
continuation. An incremental native table might be cheap, but this Python
replay makes no ranking/search-cost or latency assumption.

A separate diagnostic proposes the previous target distribution's second-ranked
token as one next token. The previous row selected the current pending token;
its alternative is only a guess for the following output. The first cycle
abstains. This checks the simplest past-logit hypothesis without adding copy
constraints, trees, model passes or cross-request memory. It is not LogitSpec.

The fixed exploratory protocol compares both context orders at caps1/4/8,
previous-second-logit at cap1, and matched ngram3/suffix5 plus ordinary controls:
14 cells per collection. All already collected CPU fit and validation datasets
are shown, keeping model/platform/phase separate. These data have already been
inspected, so this is post-hoc exploration, not another frozen validation or
permission to retune on the reserved90-prompt set. No native drafter is added
unless the work tradeoff merits a separate decision.

Before results are read,78720 small-sequence/order/cap cases match independent
reverse searches over the original observed transitions. Composition, missing
edges and bounded cycles have explicit checks. Canonical replay reuses the
corrected budget/context/EOG accounting. The existing rank-row causal guard
rejects misaligned pending tokens.

```bash
.venv/bin/python -m analysis.transition_replay \
  --collection results/verification/logit-copy-fit-20261116/qwen25-05/collection \
  --out results/local/transition-fit-05
```
