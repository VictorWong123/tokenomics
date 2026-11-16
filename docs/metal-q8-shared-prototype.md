# Shared-load Q8 prototype

`src/metal_q8_shared.metal.inc` is an unselected shader prototype. No production
CMake option or runtime dispatch uses it yet. It shares each quantized integer
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
compilation, compiler arithmetic inspection, real GPU row equality, trained
exact-output gates and M3 timing are still required before selecting the kernel.
