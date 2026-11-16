# Separating target execution and sampling

The default `verify_ms` includes target execution, backend synchronization,
logit reading and upstream greedy sampling. Historical measurements cannot
separate those costs. `TOKENOMICS_VERIFY_STAGE_TIMING=ON` adds an explicit
completed-work boundary before sampling and records two nested measurements:

- `target_compute_ms`: from target verification submission through explicit
  backend synchronization. This includes graph preparation/submission and
  completed backend work; it is not a GPU kernel-only measurement.
- `target_sampling_ms`: the existing upstream sampling/acceptance call after
  target synchronization. It includes repeated synchronization checks, host
  candidate preparation and greedy selection for all committed rows.

Both appear in detailed cycles and aggregate `execution` totals, including
minimal logging. Their sum is bounded by the original `verify_ms`; small timer
and boundary overhead remains in the combined total. Optional logit diagnostics
run before this boundary and are included in `target_compute_ms`. Do not use
logit diagnostics for timing conclusions.

The default build remains OFF. Instrumented binaries identify themselves with
`TOKENOMICS_VERIFY_STAGE_TIMING=ON` in `system_info`, so their calibration backend
identity differs. The added synchronization/timestamps can affect timing.
Compare policies with the same instrumented executable; use the ordinary
uninstrumented build for final speed comparisons. The paired short-MV study
explicitly configures this diagnostic OFF.

## M3 diagnostic build

```bash
.venv/bin/cmake -S . -B build-metal-stages \
  -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DGGML_CUDA=OFF \
  -DTOKENOMICS_VERIFY_STAGE_TIMING=ON \
  -DTOKENOMICS_METAL_SHORT_MV=OFF -DTOKENOMICS_SANITIZE=OFF
.venv/bin/cmake --build build-metal-stages -j2
mkdir -p results/local/verification-stages
build-metal-stages/tokenomics \
  --target models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --gpu-layers 99 --threads 2 --context 2048 --batch 512 \
  --policy fixed --drafter ngram --k 1 --max-tokens 128 \
  --prompt 'Continue the sequence: abc abc abc' \
  --out results/local/verification-stages/requests.jsonl \
  --cycles results/local/verification-stages/cycles.jsonl
```

This one-request example locates stages; it does not compare policy throughput.
Use existing corpus jobs and the benchmark runner for warmed paired conditions.
A high sampling share motivates host candidate/sampler work. A high completed
execution share motivates graph/kernel profiling. Neither alone proves a
specific optimization will improve throughput.

## What the saved fit timings show

A preliminary least-squares model `verify_ms = cycles*H + verified_rows*R`, fit
separately to each of the three M3 fit prompts and six existing conditions,
gives H estimates 1.47ms (code), 1.39ms (QA), and −0.79ms (summary). The negative
estimate is physically implausible as a fixed overhead, indicating that this
small noisy dataset does not identify that simple cost model reliably. No
suffix speed forecast or controller tuning uses these estimates. Direct stage
measurements on the M3 are needed.

Reproduce that diagnostic fit, including each observed value and residual:

```bash
.venv/bin/python -m analysis.verification_cost_fit \
  --out results/local/verification-cost-fit.json
```

The command audits the archived fit payloads first and also reports the CPU
operating point. It does not fit on development validation or held-out prompts.
