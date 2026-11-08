# Preliminary trained-model screen

Observed Codespace run [42452778183](https://github.com/VictorWong123/tokenomics/actions/runs/42452778183),
source `81c44cc8e731c89cc4fc357f295f70f111182b64`. Three long development
prompts, cap 64, one repetition; Qwen2.5 0.5B/1.5B Q4_K_M, two CPU threads.
All nine prompt/output-ID and stop comparisons passed. Actual Shutdown was
confirmed. This is not the repeated Pilot A/B or a controller result.

`run/` retains the timing directory from the artifact without changing its raw
observations. `summary.json` records provenance, download/hash checks and the
two rough duration extrapolations. `execution-manifest.json` describes the
original full archive: its timing members map from
`qwen25-05-15-q4/timing/` to `run/` here. The downloaded archive's 61 members were
all verified; the other members (including prompt-fit checks) remain in the
Actions artifact. No model weights, credentials, runner files or build outputs
are included.

Regenerate the point estimates, cycle costs and figure from repository root:

```bash
.venv/bin/python -m analysis.timing_screen \
  results/verification/codespace-timing-20261108/run \
  --out results/local/regenerated-timing-screen
```

Fixed4 achieved 0.501× target-only throughput and confidence achieved 0.533×
on this screen. These unfavorable observations are retained. One repetition
does not characterize timing noise, and three prompts do not establish broad
feasibility. There is no confidence interval or held-out claim here.
