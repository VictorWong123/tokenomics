# How much can request-local copying save?

This offline diagnostic computes the **minimum possible verification cycles**
for a restricted retrieval family on the three frozen development-fit prompts.
It uses archived canonical trajectories and performs no model inference. It
checks15 measured upstream policy work counts per platform (30 total) before producing
the bound. Reserved held-out prompts are not read.

The oracle can choose any earlier occurrence matching the observed suffix
(up to32 tokens), copy at most the cap's number of already observed tokens,
and truncate the copy at any length. It knows the canonical future when
selecting the occurrence and length. A backward shortest-path calculation
chooses the fewest cycles, reserving one ordinary correction/bonus token per
cycle as the runtime does. Every chosen proposal is accepted, so the bound
has no rejected rows or retrieval cost. This is **not a deployable strategy**
or a speed forecast. Cross-request retrieval and learned draft models are
outside this restricted family.

Each operating point contains192 generated tokens across three prompts:

| Operating point | Minimum match | Cap | Oracle minimum cycles | Current suffix cycles | Current suffix rejected rows |
|---|---:|---:|---:|---:|---:|
| Archived M3, 0.5B | 1 | 1 | 147 | — | — |
| Archived M3, 0.5B | 1 | 8 | 115 | — | — |
| Archived M3, 0.5B | 5 | 1 | 161 | 161 | 0 |
| Archived M3, 0.5B | 5 | 8 | 137 | 137 | 5 |
| Archived Linux, 3B | 1 | 1 | 145 | — | — |
| Archived Linux, 3B | 1 | 8 | 111 | — | — |
| Archived Linux, 3B | 5 | 1 | 169 | 169 | 0 |
| Archived Linux, 3B | 5 | 8 | 155 | 155 | 38 |

The conservative cap1 drafter already reaches this restricted oracle's cycle
count on the fit traces. Changing its search structure or occurrence tie-break
cannot save more cycles there while keeping the same minimum match and cap.
Larger caps expose more opportunities but also rejected work in the actual
causal replay. Permitting one-token matches raises the optimistic copying
limit; the oracle's perfect filtering cannot be implemented by assuming that
those weaker matches are reliable. These observations do not establish
validation performance or a latency gain.

This helps distinguish our implementation from published retrieval methods.
The [SuffixDecoding authors' method description](https://github.com/Suffix-Decoding/suffix-decoding.github.io/blob/main/index.html#L420)
uses global and per-request suffix trees, frequency-scored branching, and an
adaptive speculation budget. Its agentic workloads can reuse outputs across
requests. [SAM-Decoding](https://github.com/hyx1999/SAM-Decoding#method) uses
suffix automata and can combine retrieval with learned proposals. Our bounded,
request-local linear copy has a smaller information source and different
verification costs. Their reported speedups do not apply to this bound.

Reproduce without downloading model weights:

```bash
.venv/bin/python -m analysis.retrieval_envelope \
  --out results/local/retrieval-envelope-fresh
.venv/bin/python -m unittest discover -s tests -p test_retrieval_envelope.py
```

The implementation matches an independent breadth-first enumeration on2480
small binary trajectories. Tests also enforce the observed-token copy limit
and monotonicity under relaxed match lengths or larger caps. All90 per-prompt
oracle traces and30 aggregate cells are retained in the
[evidence artifact](../results/verification/retrieval-envelope-20261116/envelope.json).
