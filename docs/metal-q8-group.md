# Two-query Q8 threadgroups

`TOKENOMICS_METAL_Q8_GROUP=ON` is a separate, default-OFF locality experiment
for the pinned Qwen0.5B target's145MB Q8_0 output projection. It requires the
short-MV verification profile, embedded shaders, and interleaving OFF.

Each threadgroup runs the original Q8_0 routine for two consecutive query rows.
The original per-row loop and reduction helper remain unchanged. A uniform
barrier precedes shared-memory scratch reuse, and odd final batches skip the
out-of-range second query before calling the routine. The host and shader use
a shared constant for group size. Original SIMD group count, output-row tiling
and scratch size are retained; query-grid height uses ceiling division by two.
Prefills and ordinary one-row decoding use the original dispatch.

The hypothesis is that keeping consecutive query rows in the same threadgroup
improves reuse of recently read weight tiles and reduces scheduling overhead.
Dequantization still runs per query. Extra barriers and serialized query work
may cost more than locality saves; this candidate can be slower. It is an
alternative to the four-type interleaving candidate, with a different tradeoff.

Host C++ compilation, selector/odd-tail geometry checks, fixture-backed runner
analysis and default CPU suites pass. A standard public Apple ARM runner also
passes native C++ compilation, Metal3.1 shader compilation, Metal library
initialization and CPU fixture suites. Its virtual GPU lacks SIMDgroup
reduction, so actual M3 numerical/performance results remain unmeasured.
No GPU speedup is claimed from these checks.

The local runner provides the same strict controls and gates:

```bash
.venv/bin/python scripts/metal_short_mv_study.py \
  --group-q8 --screen retrieval \
  --model models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --out results/local/metal-q8-group-retrieval-01
```

This builds the original guarded reference and `build-metal-q8-group`, runs
actual GPU row comparisons and both trained exact-token gates, then measures
the126-record fit screen only if they pass. `--gate-only` omits timings.
Configuration IDs, compilation flags and marker checks distinguish the profile.
`--group-q8` and `--interleave` are mutually exclusive in both CLI and CMake.
The benchmark compares the combined short-MV/grouping candidate against the
unchanged ordinary and ngram baselines; it does not isolate grouping alone.

The shared shader-embedding refactor leaves the existing interleaved shader
byte-identical to its previously compiled/inspected input. Existing results
remain scoped to their original source commits.
