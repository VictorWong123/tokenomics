# Backend sampling and the canonical greedy contract

The adapter keeps target sampling on the host. A probe of the pinned GGML CPU
argmax finds24 selection mismatches in27 synthetic cases against the original
temp-zero/greedy chain. Release and ASAN/UBSAN agree. The ordinary384-case host
sampler equivalence check still passes in both builds.

A finite counterexample uses151936 logits: tokens0 and1 both have value7, with
all others−2. The reference selects0; GGML CPU argmax selects1. Other probes
cover separated ties, signed zeros, infinities and NaNs. Unique-maximum controls
match. This is a difference in tie/edge-case contracts, and prevents treating
unmodified backend argmax as a drop-in implementation of this repository's
strict greedy endpoint.

The pinned [host greedy sampler](https://github.com/ggml-org/llama.cpp/blob/fd27b2034905f8dadd8bd6d2cc44595f4929d391/src/llama-sampler.cpp#L1054)
retains the first maximum. Its [CPU argmax](https://github.com/ggml-org/llama.cpp/blob/fd27b2034905f8dadd8bd6d2cc44595f4929d391/ggml/src/ggml-cpu/vec.h#L1697)
updates the selected index on equal maxima. The
[Metal source](https://github.com/ggml-org/llama.cpp/blob/fd27b2034905f8dadd8bd6d2cc44595f4929d391/ggml/src/ggml-metal/kernels/misc.metal#L3)
also reduces tied partial winners using maximum candidate IDs. That Metal
observation is a source audit; no GPU selection result is claimed here.

The upstream backend temperature-zero step uses argmax and gathers one
logit/candidate before final greedy selection. It also compresses host output.
Removing that step for a GPU sampling experiment would leave full sampled logits
to copy, whereas removing the host filter saves its redundant scan. These paths
need separate correctness and performance checks. A future exact GPU sampler
would need canonical first-index/edge-case handling and measurements against
an equally optimized ordinary baseline.

No backend-sampling runtime flag is enabled or added. The optional probe reads
argmax indices directly and never uses a negative/invalid result for a gather.
A completed probe reports mismatches as diagnostic data; its zero exit status
means the diagnostic completed, not that substitution is valid.

```bash
.venv/bin/cmake --build build-cpu --target test_greedy_equivalence -j2
build-cpu/test_greedy_equivalence --argmax-probe-cpu
# On an actual supported Metal device after building that target:
build-metal-q8-shared/test_greedy_equivalence --argmax-probe-metal
```

[Raw CPU evidence](../results/verification/backend-sampler-contract-20261116/summary.json).
