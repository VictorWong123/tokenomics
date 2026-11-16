# Direct backend row coverage

`test_backend_rows --metal` compares a short batch against independent one-row
operations on the actual Metal backend, rejecting CPU fallback and every
nonfinite or non-bitwise-equal result. The local Metal study runs it before
trained-model gates. It is a model-free arithmetic diagnostic, not a throughput
benchmark or a replacement for full-model verification.

The gate now covers804 cases /4181696 value comparisons:

| Operations | Cases | Exact value comparisons |
|---|---:|---:|
| F32/F16/Q4_K/Q6_K/Q5_0/Q8_0 matrix products, including14:2 grouped-query heads | 500 | 1364480 |
| Masked softmax at multiple padded KV extents | 48 | 1053696 |
| RMS normalization, norm×scale, norm×scale+bias, SiLU, SiLU×up and split SwiGLU | 96 | 1124544 |
| Normal/NeoX RoPE at absolute positions and query/KV head counts | 160 | 638976 |

Normalization tests use widths32/896/1536/4864 and2/3/5/7/9 query rows. Activation
tests include six and seven rows at width4864, crossing the pinned unary
kernel's32768-element contiguous-specialization threshold. Normalization and
activation nodes remain unmarked, as they do in the owned model graph selector.
Their one-row references use the same normal backend fusion planner.

RoPE tests use widths64/128,2/14 heads,2/3/5/7/9 rows, positions beginning at
0/248/255/511, normal and NeoX layouts, and the Qwen frequency base1000000.
Each one-row reference receives that row's same absolute position. These do not
claim to simulate KV allocation; the separate context/split tests cover it.

Source audit found that RMS-normalization thread count depends on row width and
pipeline capacity, rather than query-row count. Its dispatch needs no new patch.
Unary specialization changes addressing at the threshold; the direct test
checks that both specializations preserve the same per-value arithmetic.

Expanded Release and ASAN/UBSAN CPU runs pass. Their success validates harness
layout and CPU arithmetic; actual Metal execution remains unmeasured. Historical
receipts retain their original500-matrix/48-softmax coverage. The new checks
must run on the M3 alongside the trained exact-output gates.

```bash
.venv/bin/cmake --build build-metal-q8-shared --target test_backend_rows -j2
build-metal-q8-shared/test_backend_rows --metal
```

Experiment source fingerprints now include verification source files and nested
CMakeLists, so editing a gate implementation during a run invalidates source
identity. Component/executable hashes remain recorded separately. Older saved
fingerprints retain the scope of their own source revision.

## Header-confirmed1.5B shapes

The next harness revision adds8960-wide FFN operations,12:2 GQA with128-wide
heads,12-head softmax/RoPE and the8960-element unary dispatch boundary between
three and four query rows. It now covers1072 cases/7508672 bitwise comparisons:
640 matrix,60 softmax,132 normalization/activation and240 RoPE cases. Linux
Release and ASAN/UBSAN execution both pass with guarded CPU and Q4 pairOFF.
[Receipts](../results/verification/backend-15-local-20261116/summary.json) bind
the exact harness and executable hashes. This extends the earlier804-case
checkpoint; it does not retroactively enlarge that checkpoint's native Apple
receipt. Native compilation and actual GPU execution remain separate checks.
