# Review handoff (in progress)

## Implemented

C++20 driver pinned to llama.cpp, upstream greedy verifier and independent/lookup
drafters, owned cache retention/rollback/lazy catch-up execution adapter,
target-only and scripted fallback/re-entry, fixed caps, upstream confidence
and proposed acceptance policies. No cost-aware scheduler is selected or
implemented before feasibility evidence. The latter is an explicit remaining
task, not an omission hidden behind a benchmark wrapper.

Policy/commit unit tests and 183 differential synthetic-transformer cases cover
zero/partial/full acceptance, correction/bonus, EOG, budget/context boundaries,
request reset, changing k, fallback and synchronization. The runner persists
balanced comparison blocks, checks exact IDs, freezes hashes/configurations and
refuses incompatible resume. Q1 paired analysis and Q2 replay/prediction use
separate prompt fit/validation. Original CC0 workload and experiment manifests
are versioned. Devcontainer and optional Metal build/CI definitions are supplied.

## Actually verified

Linux **Codex Cloud**, not Codespaces: Release CPU compilation, CTest, the
fixture artifact/analysis workflow, same-command resume and resume-fingerprint
rejection. Synthetic transformer tests are not trained-model correctness or
feasibility results. No useful-model performance numbers exist yet.

Codespaces and the user's Mac have **not** run through this chat. The direct
Codespace connection fails at the GitHub API allowlist. The Mac has no attached
execution route. The devcontainer/Metal CI definitions are not reports of
execution. See `feasibility.md` for hardware and exact observed blockers.

## Exact reproduction

In Codespaces, fetch the pushed work branch (the default branch may still be
the initial README):

```bash
git fetch origin work
git switch work
git pull --ff-only origin work
bash run.sh cpu --fixtures-only --run-dir results/local/codespace-fixtures
bash run.sh cpu --run-dir results/local/codespace-pilot
```

On macOS with Xcode Command Line Tools and Python 3.12+:

```bash
git clone --recurse-submodules --branch work https://github.com/VictorWong123/tokenomics.git
cd tokenomics
bash run.sh metal --fixtures-only --run-dir results/local/mac-fixtures
bash run.sh metal --run-dir results/local/mac-metal-pilot
```

Record the exact same project revision, copy `models/models.lock.json` between
machines, and keep each run directory separate. An identical command resumes
completed blocks. `run.sh` bootstraps as well as runs; see README for direct
commands once built. The model downloader reports blocked access instead of
substituting weights. The script times a small batch before a long pilot.

Regenerate example analysis without model downloads:

```bash
.venv/bin/python -m analysis.report results/example-fixtures --out results/local/example-analysis
```

## Remaining essential work

1. Enable model access, inspect actual candidate memory/latency and run real
   token-ID tests. Resolve any batch-induced first divergence exactly.
2. Execute separate Q1/Q2 pilots on available Codespaces CPU and Mac Metal.
   Examine noisy/ambiguous data before declaring either question answered.
3. If supported, implement the cost-aware policy with development-only frozen
   priors, exploration, hysteresis and fully charged fallback/catch-up. Otherwise
   agree a justified narrower contribution or a concrete alternative.
4. Validate exact stock-executable comparison, instrumentation overhead and
   complete resource/swap/thermal observations on the selected machines.
5. Freeze the selected contribution/baselines; run five-repeat 90-prompt held-out
   evaluation and ablations. Report uncertainty, regressions and all failures.

The supplied entry script stops at step 2's scientific decision. It does not
claim that one unattended invocation can implement a controller justified by
results that do not yet exist. The full runtime project is **not complete**.

## Evidence-based résumé wording now

“Building a portable C++ speculative-decoding runtime on pinned llama.cpp;
implemented checked KV rollback, lazy draft synchronization and target-only
re-entry, validated with 183 differential synthetic-transformer cases.”

Use “in progress.” Do not add acceleration, M3 results, model-equivalence on
trained weights, energy/cost savings, research novelty or publication claims.
