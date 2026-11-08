# Original-history CPU numerical diagnosis

Run [42523392101](https://github.com/VictorWong123/tokenomics/actions/runs/42523392101),
attempt 1, source `45ba7d6aabda5fd4fc0416a8abb0b8440737c6f8`.
This is diagnostic evidence, with no accepted performance comparison.

Both original 128-token ordinary-decoding histories and their excluded warm-ups
reproduce exactly. The executables and target match the frozen failed paired
comparison. After 36 identical output IDs, guarded chooses 323 and packed 882;
their top two target logits reverse order:

| CPU profile | Top token | Top logit | Second token | Second logit | Margin |
|---|---:|---:|---:|---:|---:|
| Guarded | 323 | 27.623024 | 882 | 27.378044 | 0.244980 |
| Packed | 882 | 27.439079 | 323 | 27.407782 | 0.031298 |

These are original incremental target histories, with zero draft proposals at
the inspected rows. Different profile arithmetic/layout changes ordinary target
logits. This does not isolate a responsible low-level operator or establish
that either profile is more accurate. It does not clear the failed cross-profile
comparison or demonstrate acceleration. Guarded remains the default.

The diagnostic subprocess completed, but the old workflow report writer then
referenced an unassigned `rows` variable. The workflow conclusion is **failure**;
`report.json` was never created. The unchanged uploaded ZIP contains the original
inner archive, all 11 payload files and the execution manifest. The verifier
allows this one exact run/attempt/artifact/digest, verifies every member and
independently reconstructs the diagnostic from the original jobs/IDs/cycles.
It records the missing report and failed automation explicitly in
[verification.json](verification.json), without altering raw evidence.

The separate lifecycle artifact confirms actual Shutdown by finalizer 128865017149.
Its lifecycle success establishes cleanup, not model correctness. Independent
checks establish the two original-history reproductions; `exact_comparisons=2`
does **not** mean the two CPU profiles agree with each other.

The report writer now has a separate diagnostic branch that needs no timing
rows or benchmark environment file. Regression checks reject changed original
jobs, inconsistent top-token claims and unapproved reportless archives. No
model rerun is needed to recover an already complete diagnostic.

The final [hosted archive check](../hosted-evidence-check-20261108) passes from
the published code and reproduces the canonical diagnostic receipt exactly.

To re-audit retained measurements without inference or downloads:

```bash
.venv/bin/python -m unittest discover -s tests -p test_profile_diagnosis.py
```

The optional synthetic two-executable test needs both local builds; its earlier
and current passing receipts are separate from the trained measurements.
