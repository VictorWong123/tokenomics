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
dispatch tests pass. The expanded four-type profile also passes native Apple
C++/Metal shader compilation and native CPU fixture suites. No Metal graph
execution or throughput result exists yet.
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

The [Thinking Machines batch-invariant operations library](https://github.com/thinking-machines-lab/batch_invariant_ops)
provides another useful comparison: it replaces matrix, normalization/reduction
and log-softmax operations to make its chosen implementation invariant across
batch sizes. This reinforces auditing the complete graph rather than GEMM
alone. Replacing both one-row and batched arithmetic with a new canonical kernel
still differs from this project's requirement to preserve its existing ordinary
reference; our candidate must pass against that reference explicitly.

## Recent Apple-specific evidence

[Yemets et al., October2026](https://arxiv.org/pdf/2610.31479), section2.3.2,
reports modest small-batch Metal gains from reusing register-resident quantized
weight reconstruction across activation rows. Its supplementary speculative
example improves at verification sizes3/5/8 but disengages at9. The main matched
chat benchmark still reports no net speculative gain. These are different MLX
models, machines and arithmetic from our pinned GGUF reference, so their ratios
are not forecasts for this codebase.

For our Q8 head, a next candidate can share integer weight loads and conversions
between two query accumulators while retaining each original eight-element
dot-product order, block-scale placement and reduction helper. Expanding each
weight to floating point before the dot product would change scale placement
relative to the original Q8 routine; algebraic equivalence is insufficient for
its bitwise contract. Compared with the current repeated-helper grouping, this
candidate targets actual redundant work. It also increases register pressure.
It needs source-derived per-thread arithmetic checks, native shader compilation
and real M3 gates before performance conclusions.

[DISCO](https://proceedings.mlr.press/v262/mamou24a.html) motivates adaptive draft
length. Its learned stopping classifier differs from our EWMA cost controller;
neither can compensate for a verifier whose extra rows remain too expensive.

## Small-model and consumer-runtime studies

[Mainardi, Sandikci and Vanschoren, EACL2026](https://aclanthology.org/2026.eacl-long.255.pdf)
compare small-model methods under greedy FP16 inference on one NVIDIA A5500.
Their prompt-lookup method outperforms their other training-free choices;
external datastore search and tree construction can erase retrieval gains.
Their Qwen1.5B target matches our prospective model size, but their hardware,
precision and verification kernels differ. Their task breakdown favors grounded
summarization/RAG over short QA, supporting separate domain reporting rather
than a universal lookup speedup claim. A12GB external datastore is especially
unattractive for the user's small-memory M3. Our request-local suffix chain
avoids that memory cost; it is not the paper's tree-based REST implementation.
The useful next test is matched local target/ngram/suffix performance after
verification batching passes, not adding an expensive external datastore.

[Chordiya, July2026](https://arxiv.org/html/2607.17283v1), sections6.1 and6.5,
reports slow few-position quantized Metal verification and batch-dependent
logits. Its best speedup uses a different MPS target; its exact greedy real-model
check uses FP32 CPU. These do not establish exact quantized-Metal generation.
The independent diagnosis supports measuring completed verification rows and
kernel transitions. Its CPU sampling arithmetic is a small share of its runtime,
so our direct-host-argmax component gain should not be expected to repair the
main batching bottleneck by itself. It remains an isolated, separately gated
inference-driver optimization, including an equally optimized ordinary control.
