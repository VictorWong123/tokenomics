# Final local CPU and offline checks

At source `95c3c8a`, all 141 Python tests pass with eight optional skips. Both
collector/guided live fixtures and the model-aware native matrix fixture are
enabled for this run. Exact test output and its SHA256 are retained.
Black accepts 108 Python files, and clang-format accepts owned C++ src/tests.
The separate fresh native-only build passes five C++ suites; see the expanded
[matrix checks](../matvec-15-local-20261116/README.md).

The offline audit rechecks all 16 retained guided/transition/priority collections,
including every saved proposal, hint and work record; all match current code.
It also rechecks 440 original learned CPU comparisons/15955 measured cycles and
the pinned projection census. These are bounded correctness/reproducibility
checks, not additional independent model samples, inference runs or speed tests.

```bash
.venv/bin/python scripts/audit_research_session.py \
  --out results/local/research-audit-01
```

No candidate M3 speedup or reserved held-out result follows from these checks.
Use the [M3 handoff](../../../docs/m3-research-handoff-20261116.md) for actual
GPU gates and paired generation measurements.
