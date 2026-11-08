# Larger-target development screen and trained snapshot checks

[Run 42461309986](https://github.com/VictorWong123/tokenomics/actions/runs/42461309986)
uses source `1f9451474a64a49ce7ca328731cfa1bbc1a3d497`, a Qwen2.5 0.5B draft
and 3B Q4_K_M target on the existing two-CPU Codespace. All 21 greedy ID/stop
comparisons pass. Every fixed-cap and confidence observation is slower than
target-only. The geometric mean ratios are fixed1 0.768, fixed2 0.723,
fixed4 0.587, fixed6 0.476, fixed8 0.439 and confidence 0.682.

The separate original-KV validation captures three target traces and checks
66 restored action outputs (including the dynamic comparator and historical
signal). We independently verified each prefix, continuation, restore ID,
file hash and cycle/cache count. Maximum checkpoint state payload is
83,503,856 bytes. This supports correctness of the new saved-state path on
these trained weights; it does not establish a predictable scheduling gain.

`snapshot-setup.json` derives setup counters from the original rows. Median
snapshot restoration is 1.09 ms (QA), 1.21 ms (code), and 5.36 ms (summary).
Each of the two prefixes per group prepares draft KV once; preparation takes
1.57–18.39 seconds depending on prefix. Every action then restores the same
prepared state. These are measured diagnostic costs nested in request timers,
not a before/after throughput experiment or deployable controller overhead.

There are three development fit prompts, one cap (64) and one repetition.
`checkpoint-validation/decision.json` explicitly reports insufficient
checkpoints. No Q1/Q2 decision, frozen baseline selection, controller result or
held-out claim follows. The estimated full Pilot A duration is 29.67 hours,
an extrapolation rather than a completed experiment. Peak screen RSS is
2.75 GiB; models fit the measured 20% headroom requirement.

The ZIP, inner archive and all 173 execution-manifest members have verified
SHA-256/byte counts. This directory retains those original members, plus the
decoded workload/finalization reports, independent verification and regenerated
screen analysis. `finalize-report.json` separately confirms actual Shutdown.
`execution-manifest.json` describes original files only; added verification,
README and `analysis/` are derived records. No model weights are included.

Reproduce the screen analysis:

```bash
.venv/bin/python -m analysis.timing_screen \
  results/verification/codespace-3b-20261108/qwen25-05-3-q4/timing \
  --out results/local/observed-3b-screen
```
