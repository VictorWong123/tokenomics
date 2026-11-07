# Retained evidence

`example-fixtures/` is a curated subset of **observed synthetic-transformer
execution**, not a trained-model benchmark. Scalar measurements are unchanged;
repetition 0 is retained and cycle-file pointers are consolidated. Its manifest
identifies the exact source revision and binary/model hashes. `complete.json`
identifies the curation. No Q1/Q2 support can be inferred from these timings.

`verification/linux-cpu/` records checks actually run on the Codex Cloud CPU
allocation: CTest, Address/UndefinedBehavior Sanitizer with leak detection,
token/commit invariants, and a terminal state-transition demo. It is not a
Codespaces or Mac result. `verification.json` records unexecuted work explicitly.

Full resumable runs belong in ignored `results/local/`. Keep those immutable
per-attempt source records and transfer their directories outside Git. The
analysis/figures here are derived from the retained example and regenerate with:

```bash
.venv/bin/python -m analysis.report results/example-fixtures --out results/local/example-analysis
```

Reproduce the portable execution and common-prefix replay checks with:

```bash
bash run.sh cpu --fixtures-only --run-dir results/local/reproduction
```

For the exact historical observations, check out the code revision in the
manifest. New runs can differ in timing; fixture IDs and correctness should
remain deterministic. Do not update source while resuming an experiment.
