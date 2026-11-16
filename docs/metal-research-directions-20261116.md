# Metal speculative verification research

The local M3 results point first to verification cost, rather than controller
selection overhead. The current single-row Metal guard restores greedy token
identity but prevents verification from amortizing graph execution and weight
traffic. A different drafter can reduce rejected work; it cannot remove that
backend constraint.

## Relevant work and what transfers

[llama.cpp PR #29869](https://github.com/ggml-org/llama.cpp/pull/29869), merged
2026-10-05, implements few-row Metal matmul kernels using simdgroup matrices.
Weights are dequantized once across multiple activation rows, and some residual
adds are fused. **This implementation is already an ancestor of our pinned
llama.cpp revision.** Updating to that PR cannot be presented as a new change.
Its published M3 Ultra measurements motivate weight reuse, but the chip, model,
quantization, and validation endpoint differ from this project's M3 experiment.
The reported output-prefix agreement and numeric backend checks do not establish
bitwise equivalence to ordinary one-row inference across all generated tokens.
Our short-MV candidate deliberately retains ordinary mat-vec reductions; that
trades away some of the shared-dequantization speedup to preserve arithmetic.

[LLM-42](https://arxiv.org/html/2601.17768v2) studies deterministic execution
through controlled verification shapes, position-consistent reductions, and
cache rollback. These are useful implementation constraints. A fixed verifier
schedule, however, can define its own canonical arithmetic: agreement within
that schedule does not by itself establish agreement with our existing one-row
target. A fixed shape of one trivially retains the reference but gives up the
batching benefit we need. This distinction must remain explicit in experiments.

[Correctness Forensics for Batch Speculative Decoding](https://arxiv.org/abs/2510.22876)
examines position, masking, and cache inconsistencies when speculative acceptance
varies across sequences. Our adapter currently handles one contiguous sequence,
so its ragged multi-sequence problem is not identical. It reinforces the need to
test rejected suffixes, rollback, physical cache slot reuse, and context-boundary
behavior, in addition to acceptance counts. Partial exact-match improvements are
insufficient to satisfy this repository's exact-token correctness gate.

## Next measurable hypothesis

A conservative optimization can change **which threadgroups run next to each
other**, while retaining the same arithmetic inside each existing mat-vec
threadgroup. For dense Q4_K/Q6_K/Q5_0/Q8_0 weights and 2–9 activation rows, interleaving
activation rows within each weight tile may increase cache reuse. A wrapper
could swap the two dispatch coordinates before calling the unchanged upstream
kernel implementation. The host grid would swap the corresponding extents.
Each logical threadgroup would still execute once with the same inputs and
reduction order.

An opt-in prototype now implements this scheduling permutation as
`TOKENOMICS_METAL_INTERLEAVE=ON`, requiring short-MV and embedded shaders.
The original mat-vec kernel body is preserved; four wrappers pass swapped grid
coordinates to its existing implementations, retaining shared-memory arguments. Host C++ compilation and
dispatch tests pass. No native shader execution or throughput result exists yet.
The cache-reuse benefit remains a hypothesis. Metal does
not promise a scheduling order that guarantees the proposed cache benefit.
The output projection can exceed cache capacity even for a small model;
intermediate layer weights may already fit, making additional reuse redundant.
Any candidate needs separate opt-in dispatch, coordinate-bijection tests, actual
Metal compilation and execution, and the complete cross-profile token gate.
Only after those pass should paired M3 measurements decide whether to retain it.
The standard hosted Apple runner can resolve compilation questions, but cannot
establish throughput on the user's M3.

No reserved held-out prompt results informed these hypotheses. Existing fit and
development-validation evidence remain explicitly separated in their reports.

To gate a native prototype after building it, use the existing trained-model
cross-profile gate with `--candidate-profile interleaved`. Its marker check
requires the interleave flag on the candidate and excludes it on the reference.
The standard short-MV study explicitly forces this extra flag OFF to keep its
comparison isolated. The hosted preflight also checks an interleaved profile.

## Actual target tensor types

A bounded header inspection of the locked public Qwen0.5B file confirms291
tensors:121 F32,133 Q5_0,13 Q8_0,12 Q4_K and12 Q6_K. The output projection is
Q8_0 with shape896×151936 (144643072 encoded bytes), while the first attention
query projection is Q5_0. FFN down-projection width is4864. The filename's
`Q4_K_M` designation does not imply K-quant storage for the896-wide matrices.
The initial two-type prototype missed the large output projection. The expanded
prototype covers all four quantized types present in this model; its host grid
preserves Q8_0's distinct grouping of output rows across SIMD groups. Direct
backend tests now include Q5_0/Q8_0 and the4864-wide FFN input.

The hosted inspection read a16MiB prefix, parsed5947741 header bytes and saved
only metadata. It did not download or hash the full491400032-byte model, so its
report explicitly marks full-model SHA256 verification false. The immutable
repository revision, total byte count, HTTP content range and prefix hash are
retained with [selected header evidence](../results/verification/hosted-metal-20261116/qwen05-header-summary.json).
