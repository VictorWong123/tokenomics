# Apple M3 Metal verification

This run used an 8 GiB Apple M3 on macOS 15.6.1 (24G90), on AC power, with two inference threads. The Release build used Apple Clang 15.0.0, llama.cpp `fd27b203`, embedded Metal kernels and GPU layer setting 99. The full two-model pilot requires 5 GiB of available RAM. The Mac reported about 1.3 GiB at setup, so the controller study used Qwen2.5-0.5B-Instruct Q4_K_M with an ngram drafter. The [Linux controller study](../codespace-controller-20261108/verification.json) used Qwen2.5-3B on two CPU threads.

[Download the two-page Word report](Tokenomics-Mac-and-Linux.docx).

## Controller results after the fixes

**Ordinary decoding was fastest on both machines.** The controller produced **8.1% fewer tokens/s on M3** than ordinary decoding and **2.9% fewer on Linux**. It beat the less efficient cap-8 ngram baseline by 12.6% on M3 and 20.5% on Linux. Both independently calibrated runs selected fixed1. The M3 used a smaller target to fit available memory. The relative comparisons below pair conditions on each machine. Absolute differences across columns include the effects of model, backend and host load.

Both grids used three development prompts for calibration and six other development prompts for validation, caps of 64 and 128, three repetitions and seven conditions. Context was 2048, batch 512, with two inference threads. All **252/252** validation records and **16/16** separately selected logging records passed on each platform. Output comparisons used exact prompt IDs, generated IDs and stop reasons against the local ordinary decoder. The 90-prompt held-out evaluation remains open.

| Controller divided by baseline | M3 Metal, 0.5B | Linux CPU, 3B |
|---|---:|---:|
| Ordinary target | 0.9191× [0.8524, 0.9780] | 0.9709× [0.9607, 0.9810] |
| Frozen fixed1 | 0.9810× [0.9238, 1.0368] | 0.9943× [0.9853, 1.0023] |
| Ngram cap8 | 1.1259× [1.0016, 1.2723] | 1.2048× [1.1271, 1.2868] |
| Acceptance heuristic | 1.0034× [0.9332, 1.0687] | 1.0272× [1.0157, 1.0403] |

Ratios above 1 favor the controller. Brackets give 95% bootstrap intervals over six distinct prompts, with both output caps kept together. Against ordinary decoding, six of twelve M3 prompt/cap cells exceeded a 10% slowdown by time per token, versus zero on Linux.

| Decode throughput, tokens/s | M3 Metal, 0.5B | Linux CPU, 3B |
|---|---:|---:|
| Ordinary target | 116.00 | 7.67 |
| Cost-aware controller | 106.62 | 7.45 |
| Frozen fixed cap 1 | 108.68 | 7.49 |
| Ngram cap 8 | 94.69 | 6.18 |
| Acceptance heuristic | 106.25 | 7.25 |
| Controller without fallback | 101.50 | 7.43 |
| Controller without probes | 103.67 | 7.48 |

Rates are geometric means of prompt/cap median decode throughput. Median warm controller completion was 1.04 s on M3 and 20.43 s on Linux. Controller peak process RSS was 646 MiB and 2,227 MiB respectively. The Mac was shared and used swap during validation. The Mac decode rates should not be interpreted as an isolated-machine ceiling.

Selecting and updating the action took 0.58 µs per cycle on M3 and 1.67 µs on Linux. Detailed logging divided by minimal logging was 0.8271× on M3 and 1.0015× on Linux. Each diagnostic used only two locally selected prompts and two repetitions. The logging result has narrower scope than the six-prompt comparison.

## Verification and evidence

The repaired Release build passed all six CTest suites, including 906 CPU fixture cases. Python discovery ran 68 tests with three optional packed-CPU tests skipped. The Metal end-to-end fixture pipeline passed 168/168 records. The 0.5B trained ngram smoke passed 28/28 on both M3 CPU and M3 Metal, exercising all six controller actions. Linux previously passed 28 records for each of its ngram and model-draft checks. The full two-model trained gate and packed CPU profile were outside this Mac run.

At source `47e7a42`, short Metal verification batches used one physical row at a time. On the original failing code prompt, fixed4 and ordinary decoding then matched on all 128 tokens. The change preserves batched prompt processing. The repaired run independently recalibrated the prior, completed 18 fit records and passed all 36 validation blocks. The prior and both performance summaries were recomputed from the archived raw data.

[Verification receipt and archive hashes](verification.json) · [Fit, replay, prior and smoke records](calibration-and-smoke.tar.gz) · [Validation records](validation.tar.gz) · [Logging records](logging.tar.gz) · [Linux controller results](../../../docs/controller-results.md)

## Original failure

At source `a03ebd3`, all 24 controller records in the 28-record smoke grid matched their ordinary-decoding references on prompt IDs, output IDs and stop reasons. The smoke covered actions 0, 1, 2, 4, 6 and 8 with synthetic priors. The same binary passed 18 development-fit requests and froze a measured prior. During validation, the third comparison block failed. On `development-code-03` at cap 128, every speculative configuration emitted 6358 at zero-based position 47. Ordinary Metal decoding emitted 13. Replaying the target and fixed4 jobs with zero offloaded layers on the same Mac produced equal 128-token sequences within CPU.

The [original records](original-metal-run.tar.gz) include the two completed blocks, failed block, fit, replay, measured prior and CPU and GPU smoke outputs. The executable SHA-256 was `1b4e11e7c10224573244c6b833f7ed696fd85e6153542b86c38ea1f0f69916db`. The run stopped at the first failing block. [Issue #10](https://github.com/VictorWong123/tokenomics/issues/10)

The Mac build also exposed a calibration error when a draft model contributed a second offload record to the backend description. The fix checks the target and device identity across both groups and retains the complete target-plus-draft identity for the measured prior. [Issue #9](https://github.com/VictorWong123/tokenomics/issues/9)

## Same-model target-only screen

On the original source `a03ebd3` and repaired source `47e7a42`, the Mac replayed the three historical Linux target-only job files, including their warm-ups. Both used Qwen2.5-1.5B-Instruct Q4_K_M from the same [model lock](../codespace-gate-20261108/models.lock.json), context 2048, batch 512, two threads and cap 64. Metal offloaded all 29 layers. Each measured request generated 64 tokens and stopped at the budget. Rates exclude model load, prompt processing and warm-up.

| Development prompt | M3 original tokens/s | M3 repaired tokens/s | Linux CPU tokens/s | Output IDs across hosts |
|---|---:|---:|---:|---|
| Code 01 | 70.67 | 57.88 | 14.96 | All 64 matched |
| QA 01 | 58.07 | 69.22 | 14.84 | First difference at position 19 |
| Summary 01 | 62.61 | 64.49 | 13.29 | First difference at position 4 |

All three Mac output sequences remained identical after the fix. These are single observations per revision on a shared desktop and a historical Codespace run at source `81c44cc`. Differences in source, hardware, and two output sequences limit cross-host throughput comparisons. The Mac used roughly 1.14–1.18 GiB peak process RSS on these requests. [Mac requests and comparison](target-1.5b/) · [Linux requests and jobs](../codespace-timing-20261108/run/)

## Reproduction

From repository root, build with `bash scripts/bootstrap.sh metal` and fetch `qwen25-05` with `scripts/download_models.py` using the Linux model lock above. Run the development grid with the following command. It requires a fresh output directory and recalibrates all six actions on the local binary.

```sh
.venv/bin/python results/verification/mac-m3-metal-20261111/run_grid.py results/local/m3-repeat
```

The opt-in trained regression uses the original failing code prompt, cap 128, fixed4 and ordinary decoding. It verifies that the target actually offloaded all 25 layers. Run it after the Metal build with the pinned 0.5B model available.

```sh
TOKENOMICS_TEST_BINARY="$PWD/build-metal/tokenomics" \
TOKENOMICS_TEST_METAL_MODEL="$PWD/models/qwen2.5-0.5b-instruct-q4_k_m.gguf" \
.venv/bin/python -m unittest discover -s tests -p test_metal_trained.py
```
