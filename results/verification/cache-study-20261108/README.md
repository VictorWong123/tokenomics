# Rejected packed-weight cache study

This controlled Cloud experiment caps packed weight chunks at approximately
128 KiB during two-to-nine-row verification, retaining the existing one-row
quantizer and GEMV arithmetic. It also permits chunking a one-thread verification
instead of forcing one entire matrix chunk. Ordinary one-row decode and larger
prefills keep their original chunk selection. The tested setting uses two threads.
This is a generated-source study, not a default/profile change or trained model.

A manufactured two-layer, width-4096 Q4_K transformer produces a known
period-nine continuation. Real upstream ngram lookup supplies its proposals;
every proposal is accepted. Each process checks nine requests: caps 0/4/8,
64 output tokens and three rotating-order repetitions. All 36 requests match
both the ordinary reference and declared continuation. No oracle is injected.

| Profile | Target-only median ms | Cap4 median ms | Cap8 median ms |
|---|---:|---:|---:|
| Packed baseline | 551.057 | 351.296 | 328.325 |
| Packed, 128 KiB chunks | 484.962 | 352.881 | 331.873 |

The proposed chunk bound does not improve the verification endpoint. It is
rejected. Ordinary-row arithmetic is unchanged, yet its timings differ too;
the cause is unestablished. Per-process results are retained rather than
attributing every difference to the intended mechanism. These controlled weights,
Cloud CPU/cache and fully accepted drafts do not establish useful language-model,
Codespace or Mac performance. The cache-bound Release build passes all four
suites, including 468 strict differential cases.

`summary.json` records hashes, per-process medians and the decision. The two
provenance records contain build/binary/source identities and hardware snapshots
collected before the first measurement of each build. `raw/` retains all accepted
records, and `source/` retains the study harness and deferred CMake helper.
The 190,040,768-byte GGUF and executables are excluded from Git.

Reproduce the fixture with `tests.make_fixtures.create` using `winner=259`,
`k_quantized=True`, `ngram_cycle=True`, `width_override=4096`. Configure a normal
Release CPU build with `GGML_NATIVE/METAL/CUDA=OFF`,
`TOKENOMICS_CPU_BATCH_INVARIANT=OFF`, `TOKENOMICS_CPU_PACKED_VERIFY=ON`, and
`CMAKE_PROJECT_tokenomics_INCLUDE` pointing at the retained CMake helper.
Build `tokenomics_cache_study` first with `TOKENOMICS_STUDY_CACHE_BYTES=0`, then
131072, retaining each executable and generated-source hash. Its arguments are
profile label, fixture path and thread count. Finish all builds before timing;
run baseline/cache/cache/baseline serially. The helper applies only to the study
build and leaves the pinned submodule unchanged.
