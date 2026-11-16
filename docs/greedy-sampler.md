# Greedy target sampler experiment

`TOKENOMICS_GREEDY_NO_TEMP=ON` removes one redundant target sampler stage.
The adapter fixes temperature to zero, dynamic-temperature range to zero,
probability output to zero and grammar to empty. Its upstream sampler chain
ends with temperature-zero filtering followed by greedy argmax. Filtering scans
the candidate array and overwrites every non-winning logit with negative
infinity; argmax then scans the same array again. Calling greedy argmax directly
selects the same first maximum under these fixed settings.

The option defaults OFF. It removes only the second-last `temp-ext` stage and
checks the pinned chain's names before doing so. Any optional model token
suppression remains earlier in the chain. The original upstream
`common_sampler_sample_and_accept_n` still owns correction, bonus and greedy
acceptance. Draft-model samplers, confidence proposals, logits and stopping
rules are untouched. Raw logit diagnostics remain available.

This is an inference-driver optimization, not a new speculative algorithm.
It may improve ordinary decoding as well as speculation. Any speed comparison
must include its optimized ordinary decoder and the saved original ordinary
decoder. No measured speedup is available yet. Candidate construction and
remaining argmax work still scan the vocabulary, so the possible total benefit
is bounded by this part of sampling cost.

The direct test compares the actual upstream filter-plus-greedy chain with
greedy alone in384 cases, using vocabularies through151936 entries, ties,
positive/negative infinities, NaNs and random repeated maxima. Within-profile
runtime tests and a separate cross-binary synthetic gate check the integration.
These checks do not replace a trained-model gate or M3 timing.

## Local M3 build

```bash
.venv/bin/cmake -S . -B build-metal-greedy \
  -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DGGML_CUDA=OFF \
  -DTOKENOMICS_GREEDY_NO_TEMP=ON -DTOKENOMICS_VERIFY_STAGE_TIMING=OFF \
  -DTOKENOMICS_METAL_SHORT_MV=OFF -DTOKENOMICS_SANITIZE=OFF
.venv/bin/cmake --build build-metal-greedy -j2
.venv/bin/ctest --test-dir build-metal-greedy --output-on-failure
.venv/bin/python scripts/check_metal_profile.py \
  --reference build-metal-reference/tokenomics \
  --candidate build-metal-greedy/tokenomics --candidate-profile greedy \
  --model models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --out results/local/greedy-gate-01
```

The trained cross-binary gate checks the sampler marker and exact IDs/stops. Build the reference with both experiments and stage timing
OFF, using identical execution settings. Do not mix this sampler experiment
into the short-MV comparison: that runner explicitly disables it for both
binaries to isolate its own change.

The runtime marker is `TOKENOMICS_GREEDY_NO_TEMP=ON`. The sampler-specific
gate requires it only in the candidate and rejects the short-MV profile.
The default context 2048 gate checks 720 candidate requests, including KV
padding boundaries. Keep original and candidate ordinary decoding as separate
conditions in any later paired performance grid; within-candidate policy ratios
do not establish improvement over the saved original decoder.

## CPU component measurement

The optional `test_greedy_equivalence --bench` includes candidate-array
construction and upstream sampler application. It alternates original/candidate
order across16 trials, with32 calls per trial and matching selection checksums.
A Cloud x86 CPU run measured median244.4µs versus158.7µs at151936 vocabulary
entries (paired component ratio1.54). At32768 entries it measured52.7µs versus
35.0µs. These inputs are synthetic and cache-warm, exclude target inference,
backend synchronization and model suppression, and do not estimate M3 speed.
The removed work applies to ordinary decoding too; it cannot establish a
speculative speedup by itself.

A standard Apple ARM hosted-VM run also passes native sampler equivalence and
160 learned-model CPU comparisons. At151936 entries its component benchmark
measures median146.6µs versus89.9µs, with median paired ratio1.69 across16
alternating trials. The ratio is the median of individual ratios, not the ratio
of the two duration medians. This is a synthetic hosted CPU result, not M3
generation throughput. [Raw pairs and summary](../results/verification/trained-arm-cpu-20261116/sampler-component-summary.json).

```bash
.venv/bin/cmake --build build-metal-greedy --target test_greedy_equivalence -j2
build-metal-greedy/test_greedy_equivalence
build-metal-greedy/test_greedy_equivalence --bench
```

## Direct raw-logit selection

`TOKENOMICS_GREEDY_DIRECT=ON` is a second default-OFF experiment. It requires
`TOKENOMICS_GREEDY_NO_TEMP=ON`; keep Metal verification experiments OFF for its
isolated comparison. It builds an owned copy of pinned `common/sampling.cpp`
without modifying the submodule. The target adapter explicitly enables its
fresh sampler only when the remaining chain is exactly greedy, with no grammar,
reasoning-budget stage, probabilities, performance counters or backend sampler.
A model suppression or logit-bias stage makes it ineligible and retains the
existing no-temperature path. Draft samplers and copied/cloned samplers do not
automatically enable this optimization.

Eligible calls select directly from the contiguous host logits using strict
first-maximum comparisons, avoiding candidate-array construction. They retain
upstream synchronization, accept/history handling, correction, bonus and EOS
logic. This private path leaves one selected candidate in the common sampler;
it must not be used by callers that consume a complete distribution or perform
stochastic rejection. Fast-math compilation is rejected. Later NaNs are ignored,
a first NaN retains token zero, and ties retain the first index, matching the
pinned host sampler. Backend GGML argmax is unsuitable: the negative diagnostic
finds24 mismatches in27 cases, including adjacent finite ties.

Execution evidence adds `TOKENOMICS_GREEDY_DIRECT=ON`,
`raw_greedy_eligible=0|1` and a per-request `raw_greedy_calls` count. The direct
gate requires eligibility and a call count between completed verification cycles
and verified rows; zero-budget requests require zero calls. A build marker
alone does not establish that the optimization ran.

Build with the flags above and add `-DTOKENOMICS_GREEDY_DIRECT=ON`; use
`--candidate-profile direct` in the trained Metal gate. Keep the original
ordinary decoder and the direct candidate's ordinary decoder in later timing
comparisons. The synthetic component measurements in
[the sampler contract](backend-sampling-contract.md) do not establish generation
throughput. The verification batching bottleneck remains separate.

For a complete local M3 comparison, the shared study runner now isolates this
sampler with guarded one-row Metal verification in both binaries:

```bash
.venv/bin/python scripts/metal_short_mv_study.py --direct-greedy \
  --screen retrieval --model models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --out results/local/direct-sampler-m3-01
```

It builds/tests the original and candidate, checks host selection, then runs
separate full trained ngram and suffix gates before timing126 paired development
records. Original and optimized ordinary decoding, ngram caps1/4/8 and suffix
caps1/4/8 are retained as distinct conditions. The analyzer requires equal
logical work for each policy across binaries. Metadata identifies a host-sampler
experiment with unchanged guarded verification. Do not combine this command
with the kernel locality options, and run studies sequentially on the Mac.

The pinned1.5B Q4_K_M CPU run43846510993 also passes160 learned comparisons
between the original and no-temperature profiles with equal logical work. Its
summary is retained in `results/verification/trained-15-cpu-20261116`. That
checkpoint predates direct-logit integration. In this small correctness corpus,
suffix cap8 uses288 verification cycles/418 rows, versus ngram cap4's269/412;
more lookahead is not automatically useful. These are work counts, not latency.
Raw records and binary hashes remain in the seven-day workflow artifact; the
compact archived summary explicitly discloses that limitation.

Native Apple ARM run43848978979 additionally passes160 learned0.5B comparisons
for the direct sampler. The gate checks eligibility and actual per-request call
counts, as well as equal output IDs/stops and logical work. This is CPU execution
on a hosted VM. The archived compact summary discloses its raw-artifact limit;
it does not establish Metal or M3 throughput.
