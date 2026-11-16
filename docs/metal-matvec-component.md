# M3 matrix component diagnostic

This diagnostic compares the plain short-MV, interleaved, grouped-Q8 and shared-Q8
candidates using synthetic quantized matrices. It needs no downloaded model.
It answers whether the changed dispatch helps selected matrix operations on
the actual Mac before spending time on a full trained-model study.

```bash
.venv/bin/python scripts/metal_matvec_study.py \
  --out results/local/metal-matrix-components-01
```

Run natively on Apple Silicon after installing `requirements.txt`. The runner
builds only the optional component executable in each existing candidate build
tree. It forces Release, native CPU code, embedded Metal and short-MV ON;
sampler/stage-timing/CPU-packing experiments stay OFF. It records configuration,
hardware, SHA256 source/executable identities and raw output. Builds never
select a CPU fallback for the Metal run.

The four shapes are Q8_0[896,151936], Q5_0[896,896], Q4_K[4864,896] and
Q6_K[4864,896], each with1/2/3/5/9 activation rows. Shapes reflect the pinned
Qwen0.5B architecture and tensor-type inventory; values are deterministic random
inputs, not learned weights. The Q8 matrix occupies144643072 encoded bytes,
matching the large output head. The float setup copy is released before timing.

For every shape, all five row counts must match separately executed one-row
operations bit-for-bit, with finite, nondegenerate outputs, before that shape's
timings begin. Odd query counts exercise grouped-Q8 tails. Across profiles and
blocks, the analyzer also requires matching input, quantized-weight and output
fingerprints. A missing gate, missing/duplicate trial, failed comparison, CPU
backend or invalid duration rejects the report.

Each cell has four warmup pairs and16 alternating batch/serial timing pairs,
with eight completed graph computations per measurement. Every computation
explicitly synchronizes the selected backend. Four blocks rotate the order
short/interleaved/grouped/shared so each profile occupies each position once.
Raw5120 timing pairs and320 gate records remain available. Summary reports
per-block medians and candidate ratios against plain short-MV. Inspect one-row
controls and block variation alongside any apparent improvement; four blocks
are not a confidence interval.

The serial control contains all independent one-row matrix operations in one
graph. Its ratio measures matrix dispatch/reuse effects and excludes per-token
context overhead, attention, KV cache, sampling and model generation. Repeated
resident inputs are cache-warm; even a head larger than cache does not reproduce
full-model cache traffic. Component speed is not a prediction of end-to-end
throughput. Follow a promising result with the actual GPU row suite and trained
cross-profile token gates in `metal_short_mv_study.py`, then paired generation
measurements. A component pass cannot replace those gates.

The portable `test_matvec_component --cpu-fixture` mode checks the harness's
allocation, view layout, quantization and comparisons using four small256×128
matrices. It records20 arithmetic gates and10240 exact value comparisons and
never records timings. CPU fixture success is not a Metal arithmetic result.

[Native Apple compilation](https://github.com/VictorWong123/tokenomics/actions/runs/43827839125)
also passes for the component executable, its small CPU fixture and the grouped
shader libraries. The hosted virtual GPU cannot run the large Metal matrices;
those results are pending on the M3.
