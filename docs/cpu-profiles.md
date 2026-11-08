# CPU execution profiles

The llama.cpp pin is `fd27b2034905f8dadd8bd6d2cc44595f4929d391`.
CMake rejects a different or dirty submodule. Owned integration generates copies
in the build directory, leaving that pinned tree untouched.

## Guarded profile: default

`TOKENOMICS_CPU_BATCH_INVARIANT=ON` disables CPU repacking, llamafile and BLAS.
It excludes tiled multiplication from verification batches of up to nine rows
and keeps tiled processing for larger prefills. This repaired a measured
batch-dependent greedy divergence; the trained 272-comparison gate and later
3B screens pass. See [the investigation](divergence-investigation.md).

## Packed verification: experimental

```bash
bash scripts/bootstrap.sh cpu
.venv/bin/cmake -S . -B build-cpu \
  -DTOKENOMICS_CPU_BATCH_INVARIANT=OFF \
  -DTOKENOMICS_CPU_PACKED_VERIFY=ON
.venv/bin/cmake --build build-cpu -j2
.venv/bin/ctest --test-dir build-cpu --output-on-failure
```

The two profiles are mutually exclusive. The experimental profile requires
`gpu_layers=0`, enables CPU repacking and disables llamafile/BLAS. Its runtime
records `TOKENOMICS_CPU_PACKED_VERIFY=ON` in `system_info`.

The verifier evaluates the pending token plus at most eight proposals. Therefore
its dense weight multiplication has at most nine input rows. The pinned packed
dispatcher processes complete input planes, so its dispatch row count equals
the quantizer's `ne11` dimension. Four checked substitutions keep their layouts
and arithmetic together:

1. Use the existing one-row activation quantizer for all verification rows.
2. Skip four-row activation interleaving through nine rows.
3. Use the native packed GEMV for every one of those rows.
4. Start that GEMV loop at row zero for small batches; larger prefills retain
   upstream packed GEMM and its matching quantization layout.

Non-packed types retain the small-batch tiled guard. This changes CPU dispatch
and quantization layout, not greedy sampling or upstream acceptance/verification.
It is not a custom GPU kernel. The current trained candidates are dense Qwen;
MoE and other backend/model combinations have not been validated by this work.

Cloud Release and address/undefined-sanitizer suites pass with leak detection,
including 468 differential cases, 30 additional minimal/checkpoint comparisons
and 18 Unicode cases. Hosted preflight also passes both CPU profiles for the
prepared paired benchmark. These fixture results establish no language-model
speedup. The separate trained gate is currently running; keep the profile
experimental until its correctness, memory and throughput evidence is checked.

## Trained validation and comparison

The fixed `kernel` experiment repeats the unchanged 272-comparison trained gate
because the CPU arithmetic path changed. It also performs 14 full-generation
diagnostic comparisons, the 18-request 3B ngram screen and original-state
checkpoint checks. Every performance block must match its ordinary reference.
The target must fit with at least 5 GiB initially available and measured RSS
below 80% of that available memory.

The prepared paired comparison keeps an ordinary guarded executable, rebuilds
the packed executable in the same build path as its trained gate and checks
its exact binary/target hashes against a frozen verification receipt. It uses
three development fit prompts, cap128, three repetitions and six conditions:

| Condition | Purpose |
|---|---|
| Guarded target, minimal logs | Ordinary reference |
| Guarded target, detailed logs | Cycle-retention overhead |
| Packed target, minimal logs | Ordinary CPU-profile effect |
| Packed target, detailed logs | Packed-profile retention overhead |
| Guarded ngram cap8, minimal logs | Verification-path reference |
| Packed ngram cap8, minimal logs | Verification-path profile effect |

The complete 54-record grid is required. Each condition's executable hash is
frozen, and all profiles must emit identical prompt/output IDs and stops within
each block. This is a component study, not full Pilot A/B or a controller result.
Minimal logging still takes stage timestamps. Its comparison measures cycle
retention/bookkeeping; serialization is outside decode and inside whole-process
latency. Noise can make estimated logging overhead negative.

## Rejected alternatives

`results/verification/kernel-study-20261108` retains the forced-all-tiled study.
Its apparent speculation ratio came from a much slower ordinary baseline;
verification itself did not improve. That configuration is rejected.

`results/verification/cache-study-20261108` retains a 128 KiB packed-weight chunk
prototype. All strict fixture checks pass, but verification does not improve.
Unexplained ordinary-row timing variation is retained rather than attributed to
the intended mechanism. This prototype is also rejected. Neither controlled
study supplies trained-model, Codespaces or Mac performance evidence.
