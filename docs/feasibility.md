# Feasibility decision — development experiments in progress

Q1 (repeatable benefit over target-only): **not established**. The completed
frozen-action confirmation does not clear the practical margin.
Q2 (predictable preferable action): **unresolved**.
Recommendation: keep guarded CPU as the reference; investigate a backend with
useful batched-verification savings before a new performance study. The packed
profile passes its within-profile gate but fails the strict cross-profile token
comparison. Original-history logits reproduce that discrepancy; no packed
speedup is accepted. The guarded-profile results do not justify a full cycle controller;
full Pilot A/B and held-out requirements remain incomplete. The user explicitly
authorized the experimental controller anyway. Its completed, independently
verified six-prompt development study is documented in
[controller results](controller-results.md): 0.9709× ordinary decoding, 0.9943×
frozen fixed1 and 1.2048× cap8 ngram. It does not reverse the feasibility decision.

## Current evidence (8 November 2026)

Codespace run [42452778183](https://github.com/VictorWong123/tokenomics/actions/runs/42452778183)
completed a nine-request trained-Qwen timing screen, passed every exact-ID and
stop check, uploaded its artifact and confirmed actual Shutdown. The downloaded
ZIP, inner archive and all 61 manifest members were hash-verified in Cloud.
The dedicated two-CPU AMD EPYC 7763 host has about 7.8 GiB RAM and no swap.
At cap 64 on three long development prompts (one per category), fixed4 achieved
a geometric mean throughput ratio of 0.501 to target-only; confidence achieved
0.533. One repetition and three prompts are insufficient to answer Q1/Q2.
See `results/verification/codespace-timing-20261108` and `docs/report.md`.

The original duration screen estimated 54.95 hours for full Pilot A. Separating
fixed prefill/load/warm-up from decode gives approximately 19.72 hours. Both are
rough extrapolations, not measured full-run duration. Run
[42454095789](https://github.com/VictorWong123/tokenomics/actions/runs/42454095789)
paused successfully after 11/108 complete blocks at source
`87c7811ed5fe24024f50782a1307ecbcef52bdee`. Its 110 exact-ID checks passed;
the ZIP, inner archive and all 636 members were hash-verified. Hosted finalize
confirmed actual Shutdown. Across the 11 observed prompt/cap cells, every
drafting configuration lost to target-only (even the best observed ratio was
0.708). This is an unbalanced partial run, with no full-suite estimate,
confidence interval, tuning choice or feasibility decision. Newer replay and
warm-up protocols require a fresh point. See the retained raw aggregates in
`results/verification/codespace-partial-20261108`.

Cloud development at that revision passes 330 differential fixture cases,
25 additional minimal-logging/checkpoint comparisons, all four Release and
sanitizer CTest suites (with leak detection), 24 lifecycle tests and the complete
168-request fixture workflow. Trained-Qwen correctness was already established
by the separate 272-comparison Codespace gate. The buffer-allocation changes
are not claimed as a measured trained-model speedup.

The larger 0.5B→3B Q4_K_M pair also fits the measured memory screen (peak
2.75 GiB) but loses at every fixed cap in
[run 42461309986](https://github.com/VictorWong123/tokenomics/actions/runs/42461309986).
All 21 exact comparisons pass. The strongest fixed-cap geometric mean ratio is
fixed1 at 0.768; confidence is 0.682. This remains a three-prompt, cap-64,
one-repeat screen. It does not clear either feasibility gate. The rough full
Pilot A extrapolation is 29.67 hours on this CPU.

That run additionally validates original-state snapshots on trained weights:
three captures and 66 canonical action checks pass. The ZIP, inner archive and
all 173 members were verified; checkpoint hashes, exact prefixes/continuations
and cache/commit accounting were independently checked. Snapshot payload peaks
at 83,503,856 bytes. Three fit prompts are insufficient for Q2 validation.
Hosted finalize confirms actual Shutdown. Raw evidence and regenerated figures
are in `results/verification/codespace-3b-20261108`.

The cheaper mechanism was measured in
[42461982414, attempt 2](https://github.com/VictorWong123/tokenomics/actions/runs/42461982414/attempts/2).
All 18 exact comparisons, three captures and 66 snapshot action checks pass;
all 134 archive members were hash-verified and actual Shutdown confirmed.
Ngram fixed1 reaches a geometric mean ratio of **1.0113** to target-only;
fixed2/4/6/8 reach 0.9429/0.9205/0.8402/0.7821. Actual fixed1 proposals and
accepted counts are 6/5 (QA), 14/14 (code), 18/14 (summary). Peak RSS is
2.18 GiB. Attempt 1 failed startup before inference; the unchanged lifecycle
succeeded on retry. Raw records are in `results/verification/codespace-ngram-20261108`.

The 1.1% shortest-cap point is below the frozen 5% practical margin and cannot
distinguish useful benefit from noise. The predeclared next check freezes that
action from these three fit prompts and uses nine distinct development prompts,
caps 128/512 and three repetitions against ordinary target and variable-length
ngram cap8. `configs/ngram-confirmation-v1.json` hashes the fit evidence;
`scripts/confirm_ngram.py` rejects drift, retuning and partial-grid analysis.
It uses bounded resumable invocations. This targeted confirmation does not
replace the full Pilot A/B or held-out protocol and cannot itself unlock a
cycle controller. The full ngram Pilot A extrapolation is 12.25 CPU hours.

## Completed frozen-action confirmation

Run [42466570831, attempt 2](https://github.com/VictorWong123/tokenomics/actions/runs/42466570831/attempts/2)
completed all 54 blocks and 162 exact-ID/stop comparisons on nine distinct
development validation prompts, caps 128/512 and three repetitions. Cap 1,
frozen from the separate three-prompt fit screen, achieved a geometric mean
throughput ratio of **0.9846** to ordinary target-only, with prompt-clustered
95% interval **[0.9721, 1.0019]**. Cap 8 achieved **0.7727 [0.7395, 0.8063]**.
The frozen 5% practical-gain criterion is false; the selected action is not
retuned from this validation.

All 927 manifest members, completed-block hashes, request IDs/stops, cycle/cache
accounting and regenerated ratio/interval statistics pass independent hosted
verification. The connected GitHub API supplied the retained finalizer report,
confirming actual Shutdown. Peak process RSS is 2.18 GiB. Raw bytes are retained
in `results/verification/codespace-confirmation-20261108/attempt-2`, with the
canonical verification receipt and lifecycle evidence alongside them.

This rejects a practical-gain claim for the frozen action on this validation
point. It is not a completed 18-prompt Pilot A, a Q2 decision, an all-cap or
all-platform impossibility result, a controller experiment or held-out evidence.
The separate packed CPU profile later passes its trained gate, then fails
cross-profile greedy equality in the paired comparison. The independently
checked original-history diagnosis and its reporting limitation are documented
in [CPU profiles](cpu-profiles.md).

## Historical foundation and access blockers (7 November 2026)

2026-11-07, Codex Cloud Linux x86_64; AMD EPYC 9V74, five exposed logical CPUs,
cgroup CPU quota four cores, 16 GiB memory limit, no swap or GPU. GCC 14.2.0,
CMake 3.31.6, Ninja 1.11.1, Python 3.12.14, NumPy 2.3.5. Release CPU build with
GGML_NATIVE=OFF, GGML_METAL=OFF, GGML_CUDA=OFF. No machine is called an M3 here.

The C++ runtime passed 183 differential fixture cases and the core policy/commit
checks. A 168-request fixture experiment (24 complete blocks) passed exact output
ID comparison and generated analysis artifacts; resume was tested. These use
tiny, controlled/random F32 transformer weights generated locally. They are
correctness evidence only, not useful language-model generation or feasibility.

Address/undefined-behavior sanitizer checks with leak detection passed.
The documented bootstrap was also exercised in a fresh virtual environment,
which caught and fixed missing GGUF-helper dependencies before final handoff.

Real model metadata/download probe:

```bash
python scripts/download_models.py --model qwen25-05
```

Observed failure: `Tunnel connection failed: 403 Forbidden` for
`https://huggingface.co/api/models/Qwen/Qwen2.5-0.5B-Instruct-GGUF?blobs=true`.
The configured cloud allowlist lacks Hugging Face. No trained weights or pilot
measurements were produced. Domains need enabling in environment network
settings; do not send credentials in chat.

## Separate environments at the original handoff

* **Codex Cloud Linux CPU:** source build, synthetic state/differential tests,
  resumable request/analysis pipeline executed.
* **GitHub Codespaces:** user created `orange-space-giggle-gwpgvr4rvg7fgv9`.
  Direct `gh codespace ssh -c ...` fails before SSH at an `api.github.com`
  metadata request with Forbidden. Neither build nor model tests have run there
  through the original chat. This historical route was replaced by the verified
  dedicated Actions lifecycle; current execution is documented above.
* **User's Mac:** no authorized shell/terminal connection is currently attached.
  Chip/RAM/power state have not been inspected. Metal configuration and commands
  are supplied, but no Metal execution or M3 result exists.

## Bounded investigation prepared

`configs/models-v1.json` justifies two candidate pairs using available RAM,
then one upstream ngram mechanism if both independent pairs are unprofitable.
`configs/pilot-v1.json`, `bench/run.py` and `bench/replay.py` implement the
separate Pilot A and B procedures. Run independently on each platform:

```bash
bash run.sh cpu --run-dir results/local/linux-pilot
bash run.sh metal --run-dir results/local/mac-metal-pilot
```

Results are local, ignored, resumable, and transferable. A prior from one
platform is not evidence for a different machine. The script stops at a
scientific gate and records its reason. The separately authorized experimental
controller is now implemented; its full held-out validation remains future work.
