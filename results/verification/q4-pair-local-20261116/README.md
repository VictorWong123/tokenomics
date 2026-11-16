# Q4 pair component experiment

These are synthetic matrix measurements on the Codex Cloud AMD EPYC CPU, not
trained-model or Codespaces throughput. The opt-in Q4_K AVX2 kernel shares weight
unpacking across two activation rows and preserves the pinned dot-product's
per-row arithmetic. Ordinary single-row runtime dispatch is unchanged.

| Matrix (outputs × width) | 2 rows | 3 rows | 5 rows | 7 rows | 9 rows |
|---|---:|---:|---:|---:|---:|
| 4,096 × 4,096 | 1.236× | 1.134× | 1.157× | 1.164× | 1.174× |
| 32,768 × 2,048 | 1.590× | 1.355× | 1.277× | 1.195× | 1.180× |

Each number is the median of 16 paired baseline/candidate timing ratios, with
alternating execution order and a warm-up of each path. All output float bits
match in every accepted record. The one-row controls (0.960× and 0.995×) are
retained too; their harness loop order differs, while actual one-row runtime
execution remains on the original implementation. No compilation or other
inference ran during these measurements. They use one benchmark thread.

The initial harness forgot `ggml_cpu_init`, leaving a CPU half-conversion table
uninitialized and all results zero. Its timings and checks are invalid. The
corrected harness initializes the backend and rejects zero checksums. The
invalid run's raw hashes and reason remain in `invalid-initial-harness.json`.

`source.json` binds the uncommitted branch prototype's kernel, harness and
generated-source bytes; `binaries.json` records executable hashes.
`environment.json` was captured before the valid measurements. The two matrix
JSONL files retain all 192 paired measurements, and `summary.json` is derived
from them. Release correctness logs are separate from component timing.
The paired fixture grid uses a Q4 period-nine model and explicitly requires
nonzero drafted/accepted tokens. An earlier random fixture failed that coverage
requirement; its failure log is retained separately from the passing grid.
All seven address/undefined sanitizer suites pass with leak detection, taking
395.03 seconds; `pair-asan-ctest.txt` retains the complete suite summary.

Reproduce after configuring a guarded Release build with
`-DTOKENOMICS_CPU_Q4_PAIR=ON` and the documented portable CPU switches:

```sh
.venv/bin/cmake --build build-q4-pair --target bench_q4_pair test_q4_pair -j4
build-q4-pair/test_q4_pair
build-q4-pair/bench_q4_pair 4096 4096 16
build-q4-pair/bench_q4_pair 2048 32768 16
```

Finish compilation before timing. End-to-end trained correctness, memory and
throughput require the separately declared 272-case gate and paired study.
