# Controller development measurements

Bounded, frozen development engineering comparison; uncertainty resamples distinct prompts, not repetitions. No full Pilot A/B or 90-prompt held-out result.

Frozen fixed comparator: `fixed1`. Synthetic fixtures: `False`.

| Policy | Decode tokens/s¹ | Median warm ms | Peak RSS MiB | Decision + update / generation |
|---|---:|---:|---:|---:|
| cost-aware | 7.258 | 40177.72 | 2227.5 | 0.001% |
| cost-aware-detailed | 7.269 | 40191.15 | 2226.3 | 0.001% |
| fixed1 | 7.230 | 40343.75 | 2223.5 | 0.001% |
| target | 7.560 | 39350.17 | 2222.4 | 0.001% |

¹ Geometric mean of prompt/cap median throughput. Full probe, synchronization and update work remains charged. RSS is lifetime process peak, not policy memory savings.

| Comparison | Throughput ratio | Prompt-bootstrap 95% interval |
|---|---:|---:|
| cost-aware/target | 0.9601 | [0.9163, 1.0059] |
| cost-aware/fixed1 | 1.0039 | [0.9782, 1.0303] |
| cost-aware/cost-aware | 1.0000 | [1.0000, 1.0000] |
