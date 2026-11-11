# Retained evidence

This directory separates observed execution evidence from derived reports and
local experiments. The completed controller result is a bounded Linux CPU
**development** study, not a held-out or Mac result.

## Current engineering checkpoint

| Evidence | What it establishes |
|---|---|
| [Controller study](verification/codespace-controller-20261108/verification.json) | Complete 252-record validation grid, 16 logging records, frozen calibration, exact IDs/stops and actual shutdown |
| [Apple M3 Metal tests and Linux comparison](verification/mac-m3-metal-20261111/README.md) | Native fixture checks, trained greedy output comparisons and target-only decode measurements |
| [Controller results and figures](../docs/controller-results.md) | Ratios, uncertainty, ablations, overhead, work counts and limitations derived from that study |
| [Four-page PDF](verification/engineering-report-20261108/tokenomics-engineering-report.pdf) | Reproducible engineering report; generator/receipt/figure hashes are in `report-provenance.json` beside it |
| [Frozen-action confirmation](verification/codespace-confirmation-20261108/verification.json) | Separate nine-prompt fixed1 result; no practical ordinary-decoding gain |
| [Original trained gate](verification/codespace-gate-20261108/independent-verification.json) | Original 272-comparison trained correctness evidence |
| [Packed trained gate](verification/codespace-packed-controller-20261108/verification.json) | Experimental profile's within-profile correctness/replay gate; no paired speedup claim |
| [Failed paired CPU comparison](verification/codespace-cpu-profiles-20261108/README.md) | Preserved cross-profile token mismatch; no complete performance summary |
| [Original-history diagnosis](verification/codespace-profile-diagnosis-20261108/README.md) | Reproduced outputs and reversed ordinary target logit rankings; post-inference reporting failure retained explicitly |
| [Hosted archive check](verification/hosted-evidence-check-20261108/README.md) | Independent recheck of the five frozen experiment archives; starts no inference |
| [Guarded local checks](verification/controller-local-20261108/verification.json) / [packed local checks](verification/controller-packed-local-20261108/verification.json) | Historical Release/sanitizer/differential fixture evidence; not language-model performance |

Controller/ordinary throughput is 0.9709×, controller/frozen-fixed1 is 0.9943×
and controller/cap8 is 1.2048×. The [result document](../docs/controller-results.md)
contains the confidence intervals, exact operating point and unfavorable cells.
Full Pilot A/B and the 90-prompt held-out evaluation remain incomplete.

## Layout and integrity

- `verification/<point>/attempt-N/` retains immutable raw payloads and execution
  manifests. A canonical `verification.json` records the checked point. Failed
  grids use attempt-specific failure receipts where applicable.
- `derived/`, `analysis/` and PDFs are derived artifacts, distinct from raw
  execution records. Recompute new views into `results/local/`.
- `example-fixtures/` contains actual synthetic-transformer observations for
  no-download analysis. It is not a trained-model benchmark: scalar values are
  unchanged, repetition 0 is retained and cycle-file pointers are consolidated.
- Earlier directories such as `verification/linux-cpu/`, timing screens,
  partial pilots and rejected kernel/cache studies remain historical evidence.
  Their exact manifest/source and documented scope govern interpretation.
- `results/local/`, model weights and build trees are ignored by Git.

Do not rename, delete, reformat or combine existing raw payloads, hash-bound
configs or canonical receipts during repository cleanup. Frozen tests and
calibration provenance refer to their paths and byte hashes. GitHub artifact
expiry does not remove the checked-in raw files and verified shutdown records.
For exact historical reproduction, check out the manifest's source revision.

## Inspect without trained inference

After [development setup](../docs/development.md):

```bash
.venv/bin/python scripts/demo_controller.py --max-cycles 12
.venv/bin/python -m analysis.report results/example-fixtures \
  --out results/local/example-analysis
.venv/bin/python -m analysis.controller \
  results/verification/codespace-controller-20261108/attempt-2/validation \
  --out results/local/controller-analysis --no-plots
```

The [evaluation guide](../docs/evaluation.md) documents supplemental figures and
PDF regeneration. New runs require their own immutable directory when source,
executable, models or settings change. Never resume an old point with changed code.
