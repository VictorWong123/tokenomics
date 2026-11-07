# Synthetic fixture pipeline demonstration — NOT feasibility evidence

Selected development fixed setting: `fixed2`.

| Configuration | Decode tokens/s (geometric mean of cell medians) | Median warm latency ms | Peak lifetime RSS MiB |
|---|---:|---:|---:|
| acceptance | 14068.29 | 5.39 | 37.9 |
| confidence | 15994.85 | 4.93 | 37.9 |
| fixed2 | 13722.69 | 5.65 | 37.9 |
| fixed4 | 13557.11 | 6.32 | 37.9 |
| resident | 22496.03 | 3.58 | 37.9 |
| target | 22200.31 | 3.47 | 37.9 |
| upstream-style | 11122.00 | 7.32 | 37.9 |

Paired uncertainty resamples distinct prompts with both output caps together. Timing repetitions are not independent prompts. Model load, request setup, prompt processing and complete generation are retained separately. No memory, energy or dollar-cost reduction is inferred from throughput.

Q1 validation: {"fit_selected_fixed": "fixed4", "ratio": 0.5801458137075874, "ci95": [0.5576249826087208, 0.5974237165308773], "prompts": 3, "prompt_cap_cells": 6, "wins": 0, "ties": 1, "losses": 5, "more_than_10pct_slower": 5, "worst_ratio": 0.32467561002799944}

Q2 is a separate checkpoint replay investigation; this table does not establish predictable within-generation adaptation.
