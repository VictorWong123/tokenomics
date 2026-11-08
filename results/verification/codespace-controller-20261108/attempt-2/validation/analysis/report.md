# Controller development measurements

Bounded, frozen development engineering comparison; uncertainty resamples distinct prompts, not repetitions. No full Pilot A/B or 90-prompt held-out result.

Frozen fixed comparator: `fixed1`. Synthetic fixtures: `False`.

| Policy | Decode tokens/s¹ | Median warm ms | Peak RSS MiB | Decision + update / generation |
|---|---:|---:|---:|---:|
| acceptance | 7.252 | 20838.80 | 2225.3 | 0.001% |
| cost-aware | 7.450 | 20432.58 | 2227.2 | 0.001% |
| dynamic | 6.183 | 23849.30 | 2227.5 | 0.001% |
| fixed1 | 7.492 | 20208.76 | 2223.4 | 0.001% |
| no-fallback | 7.427 | 21431.35 | 2226.3 | 0.001% |
| no-probes | 7.477 | 20263.34 | 2224.1 | 0.001% |
| target | 7.673 | 19981.82 | 2222.3 | 0.001% |

¹ Geometric mean of prompt/cap median throughput. Full probe, synchronization and update work remains charged. RSS is lifetime process peak, not policy memory savings.

| Comparison | Throughput ratio | Prompt-bootstrap 95% interval |
|---|---:|---:|
| cost-aware/target | 0.9709 | [0.9607, 0.9810] |
| cost-aware/fixed1 | 0.9943 | [0.9853, 1.0023] |
| cost-aware/dynamic | 1.2048 | [1.1271, 1.2868] |
| cost-aware/acceptance | 1.0272 | [1.0157, 1.0403] |
| cost-aware/cost-aware | 1.0000 | [1.0000, 1.0000] |
