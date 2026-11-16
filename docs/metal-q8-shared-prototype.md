# Shared-load Q8 prototype

`src/metal_q8_shared.metal.inc` is an experimental, default-OFF shader. It shares each quantized integer
weight load/conversion and scale load across two query rows. Each query keeps
the original eight-element dot-product order, multiplies its completed dot by
the block scale, and calls the unchanged pinned reduction helper. A uniform
barrier separates the two uses of shared reduction scratch. An odd last query
skips the second row. This differs from `TOKENOMICS_METAL_Q8_GROUP`, which calls
the whole original routine twice and repeats its weight work.

A dedicated host check compiles the **actual original shader function extracted
from the clean pinned submodule**, plus the candidate function source, as C++.
Only address-space syntax and the final Metal reduction helper are replaced.
The replacement captures each thread's partial sums; it does not emulate SIMD
reductions or GPU synchronization. Thus this checks the source arithmetic and
index coverage, not the final GPU output or compiler equivalence.

```bash
.venv/bin/cmake -S tests/metal-q8-arithmetic -B build-q8-arithmetic \
  -DCMAKE_BUILD_TYPE=Release
.venv/bin/cmake --build build-q8-arithmetic --parallel 2
.venv/bin/ctest --test-dir build-q8-arithmetic --output-on-failure
```

Two builds disable contraction or allow native FMA contraction. Each checks2016
cases and3311616 bitwise partial sums across widths32–4864,2–9 query rows,
1/2/4 SIMD groups, two output sizes, padded/contiguous rows and random,
cancellation-heavy and tiny activations. Scales include zero, signed zero,
small half values and the largest finite half. Every expected lane/query must
write once; scratch-barrier counts must agree with complete query pairs.

Both Release variants and ASAN/UBSAN variants pass locally. The contracted x86
binary contains actual FMA instructions. A negative control changes only the
candidate weight conversion by one; the checker rejects it, then the original
source and executable are restored byte-for-byte. No component performance is
measured by this test.

The source-only Metal diagnostic can append this prototype while leaving the
interleaved/native runtime inputs unchanged. The compiler-IR report explicitly
lists additional functions outside its semantic comparison. Native shader
compilation and source-derived partial checks also pass on Apple ARM. Its AIR
contains a shared integer conversion feeding two independent accumulators;
the operations carry `fast` flags, so later GPU compilation remains relevant.
All262 original functions and four interleaved wrappers still match their
separate reference. The new shared helper/kernel are explicitly outside that
semantic comparison. [Native diagnostic](../results/verification/q8-shared-partials-20261116/native-ir-comparison.json).

An opt-in runtime integration now requires `TOKENOMICS_METAL_Q8_GROUP=ON` and
`TOKENOMICS_METAL_Q8_SHARED=ON`, in addition to Apple Metal, embedded shaders
and short-MV. The existing selector and ceiling-divided two-query grid remain
the same; only the named kernel changes from `_grouped2` to `_shared2`.
The runtime adds a distinct marker, and gates reject mixed sampler/kernel
profiles. Ordinary one-row decode and prefills keep the original dispatch.

The runner builds the original guarded reference, executes the actual GPU row
suite and trained exact-output gates, then measures only if those pass:

```bash
.venv/bin/python scripts/metal_short_mv_study.py \
  --share-q8 --screen retrieval \
  --model models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --out results/local/metal-q8-shared-retrieval-01
```

`--gate-only` skips timings. The model-free matrix component runner also includes
the new profile alongside the serial grouped control, separating actual shared
weight work from grouping alone. Real GPU equality, trained Metal output
agreement and M3 throughput remain unmeasured.
