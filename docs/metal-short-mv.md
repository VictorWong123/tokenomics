# M3 Metal short-batch experiment

This research-branch experiment is **off by default and not yet validated on a
Metal device**. The intended machine is the user's Apple M3. Linux compilation
and CPU fixture checks cannot establish Metal correctness or speed.

The existing Metal guard splits verification into one-row microbatches. It
repaired a Qwen2.5 0.5B divergence on `development-code-03` at output token 47,
but serializes work that speculative decoding needs to amortize. The opt-in
`TOKENOMICS_METAL_SHORT_MV=ON` removes that split and routes two-to-nine-row
ordinary matrix multiplications through the existing mat-vec dispatcher. Each
GPU threadgroup retains the ordinary per-row arithmetic. It also disables
MMA/MM selection and runtime MMA+ADD fusion for these shapes. The structural
fusion planner remains shape-independent, as required upstream.

The integration generates owned copies of `ggml-metal-common.cpp` and
`ggml-metal-ops.cpp`; it does not modify the upstream pin or Metal shaders.
Ordinary one-row dispatch and batches above nine rows remain upstream paths.
The selector also affects short prefills and attention matmuls. Attention KV
padding, other fused operations and the rest of the graph may still cause a
whole-request mismatch. This is a hypothesis to test, not a proof of batch
invariance. It shares a graph/launch across rows but does not introduce a fused
weight-reuse kernel, so it may remain slower than ordinary decoding.

## Local command

Use the research branch `research/speculative-throughput-20261116` and native
arm64 Python (not Rosetta). Xcode Command Line Tools, the clean pinned submodule,
and the requirements environment are needed. Reuse the existing pinned model:
`qwen2.5-0.5b-instruct-q4_k_m.gguf`, SHA256
`74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db`.
No draft model is loaded, to keep the experiment suitable for an 8 GiB M3.

```bash
git submodule update --init --recursive
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/metal_short_mv_study.py \
  --model models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --out results/local/metal-short-mv-01
```

The command builds separate guarded and candidate binaries with two compiler
jobs, runs their test suites, and starts with the original M3 failure case.
It then compares the candidate with the unchanged ordinary decoder across
330 requests: fixed caps 1/2/4/6/8, scripted fallback/re-entry, both cache modes,
repeated requests, minimal logs, token budgets and context boundaries. Marker,
GPU device and complete layer-offload checks reject silent CPU fallback. The
gate retains raw requests, cycles and first mismatch and stops immediately on
failure. `--gate-only` omits performance measurements.

Only a passing gate enables the 72-record screen: three development fit prompts,
cap128, three repetitions and eight conditions. It compares each candidate
fixed policy with the same guarded policy, candidate ordinary decoding and
unchanged ordinary decoding. Paired policy work counters must agree exactly.
The runner uses the existing blocked/randomized ordering, frozen executable
hashes and independent ordinary references. Analysis reports generation and
warm-request ratios and aggregate target-prefill costs. The three prompts are
a component screen; they do not support a held-out or controller claim.

Keep the Mac plugged in, close heavy applications and let it cool before the
screen. The runner captures before/after system, memory and thermal metadata.
Do not edit source or rebuild the binaries during a run. Every run requires a
fresh output directory, preserving failed attempts. Review logs for private
local paths before sharing; never upload credentials or model weights.

## Checks available without a Mac

The generated host C++ sources compile on Linux. A host-only predicate program
checks 320 combinations across F32/F16/Q4_0/Q4_K/Q6_K, row counts 1–32 and one or
four heads. CPU fixtures test the gate's complete comparison grid and fail-fast
retention. These checks exercise integration and orchestration only. The Mac
run above remains necessary, including when these checks pass.

Reproduce the Linux host check after building `build-cpu`:

```bash
.venv/bin/cmake -S tests/metal-host -B build-metal-host
.venv/bin/cmake --build build-metal-host -j2
.venv/bin/ctest --test-dir build-metal-host --output-on-failure
```

This target deliberately links only the host predicates; unused GPU runtime
functions are discarded. It is not a usable inference backend.

New gate runs write `gate/complete.json` with hashes of the manifest, jobs,
outputs, cycles, logs and pass status. Verify a saved gate without inference:

```bash
.venv/bin/python scripts/verify_metal_gate.py results/local/metal-short-mv-01/gate
```

This checks accidental payload changes, execution settings, saved exact outputs,
cycle accounting and comparison coverage. It does not recreate GPU execution or
authenticate an artifact whose payload and receipt were both replaced. Older
gates without this receipt need to be rerun to use this check.

## Direct matrix-row check

The local runner first builds `test_backend_rows` for the candidate and executes
it with `--metal`. It compares each row of a batched matrix multiplication with
an independent one-row multiplication using identical weights and activations.
The grid includes F32/F16/Q4_K/Q6_K, short batches through nine rows, widths
64–1536 where supported, and both independent and broadcast weight heads.
Outputs must be finite, nondegenerate and bitwise identical. A first mismatch
prints its shape and values and prevents the trained gate/timing stages.

```bash
.venv/bin/cmake --build build-metal-short-mv --target test_backend_rows -j2
build-metal-short-mv/test_backend_rows --metal
```

Linux can run the same harness with `--cpu`; that checks the harness and CPU
backend only. `--metal` rejects a build without Metal and never falls back to
CPU. This test covers matrix operations, not the complete attention graph, KV
padding, normalization or end-to-end model behavior. The trained gate is still
required even when direct row comparisons pass.
