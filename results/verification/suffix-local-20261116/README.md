# Suffix CPU checks and offline fit replay

The opt-in `suffix` drafter is checked on CPU fixtures, not a Metal device.
Release passes seven suites. Address/undefined sanitizers with leak detection
pass suffix, fixture setup and runtime suites (the existing sanitizer build
also enables the x86 Q4 pair experiment). There are 9,216 retrieval cases and
1,134 differential runtime cases. Source and binary hashes are retained.

`proposal-replay.json` contains all 300 offline comparisons across two archived
fit points, ten proposal strategies, five caps and three prompts. It reproduces
30 measured upstream aggregate work counts before evaluating alternatives.
The archives have no detailed cycle records, so zero cycle-level checks are
claimed. The threshold-five/cap-one proposal was selected on these fit data;
its zero rejected proposals are not a held-out accuracy or speed result.

No trained inference ran in Cloud: the pinned model download was rejected by
the environment network proxy. The local M3 gate and screen remain necessary.
See [design and commands](../../../docs/suffix-drafter.md).

Reproduce the offline result in a fresh directory with
`python -m analysis.proposal_replay --out results/local/proposal-replay-02`.
The output's source identity reflects the current checkout; the per-source
identities, proposal counts and totals should match this retained result.
