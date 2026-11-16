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
The default context2048 gate checks510 candidate requests, including KV
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

```bash
.venv/bin/cmake --build build-metal-greedy --target test_greedy_equivalence -j2
build-metal-greedy/test_greedy_equivalence
build-metal-greedy/test_greedy_equivalence --bench
```
