# Frozen feasibility protocol v1

## Scope and platforms

Policy code uses no Metal/CUDA APIs. Build and compare supported llama.cpp
backends on each actual machine. Codespaces is an allocated Linux VM/container;
its throughput is not a result for a typical desktop or a Mac. The current
executed host is Codex Cloud, not Codespaces. Record cgroup CPU and memory
limits in addition to exposed hardware. The Codespace trained-model correctness
route is now verified; development performance execution is a separate task.
Develop/check changes in Cloud, push checked stages to main, then run the
measurement workload in Codespaces. Mac tests must record chip, RAM,
macOS, power/thermal state and Metal build. No NVIDIA GPU is required.

Within every comparison: identical target file SHA-256, quantization, prompt
IDs, greedy sampler, stop rules, logical context, F16 KV, batch/ubatch, backend
and threads. `max_tokens` counts sampled EOG. Model/resident memory differences
are explicit. Cross-platform differences must be reported as configuration
differences; controller gains are paired within one platform only.

## Workload provenance

`bench/prompts-v1.jsonl` contains 40 development and 90 held-out original,
AI-assisted synthetic prompts under CC0-1.0. Held-out contains 30 Q&A, 30 C++
completion and 30 summarization prompts, balanced short/long within category.
Summary source documents are embedded. Generator source, prompt IDs and SHA-256
are versioned. They contain no private/public-source copyrighted text.
This is a controlled engineering workload, not an external benchmark or quality
evaluation. Synthetic summary templates share structure and are correlated;
confidence intervals over these prompts do not establish generalization to
natural documents. Real prompt token lengths are recorded by the selected target
before experiments. The script refuses a prompt+cap that exceeds context.

ChatML formatting is literal and frozen for Qwen2.5. Automatic templates and
reasoning-budget logic are disabled. Both prompt IDs and output IDs must match.
The manifest must change if a different tokenizer/model family is selected.

## Pilot A: useful speculation

18 development prompts, six per category, output caps 128/512, three repeats.
Every prompt/cap/repetition is a comparison block. A seeded base permutation
is rotated across shuffled blocks to balance configuration order; one request
is active at a time. Each trial starts a process, loads appropriate models,
runs an excluded 16-token warm-up, clears request state and measures the real
request. Initialization, warm-up and whole-process cost are retained separately
from warm generation. This avoids a resident draft contaminating target-alone
peak RSS. The resident-draft control warms both models through an excluded
speculative request before measuring target-only; merely mapping the draft file
would not establish physical weight residency. Load/file-cache variability
remains visible, not silently discarded.

The initial timing batch is three longer development requests (one per category) at cap 64 with target,
fixed4 and confidence; its wall time predicts an approximate full run budget.
Pilot A has 1,080 measured requests per pair (18×2×3×10), plus warm-ups and
replays. EOS and hardware can make a CPU run take hours. The script prints its
measured estimate before proceeding; it does not promise an overnight finish.

Configurations: target-alone, target with resident draft, upstream-style cache
replay with cap4, retained-cache fixed caps {1,2,4,6,8}, pinned upstream
confidence stopping at cap8/p_min=.5, and the proposal's acceptance rule.
Confidence uses upstream top-10 normalized draft confidence; it is not a
literal reproduction of every Mamou et al. setting. The upstream-style path
reuses the actual common drafter and verifier but is a control inside this
driver, not the stock executable; an exact stock-executable benchmark remains
an additional validation task before final comparison claims.

The global fixed action maximizes geometric mean prompt/cap median throughput
on development data only. Q1 feasibility additionally selects a fixed action
on prompts 00–02 in each category and tests it on 03–05, preserving prompt-level
separation. Require >5% ratio and bootstrap lower bound >1.0. This conservative
decision rule does not make exploratory model selection confirmatory research.
Investigate ambiguous outcomes with targeted repeats; do not label all failures
of this strict gate as proof that speculation cannot help.

## Pilot B: predictable scheduling opportunity

Use ordinary target greedy traces at cap512/repetition0. At generated prefixes
32,96,224,400, if nine later trace tokens exist, replay each action {0,1,2,4,6,8}
from the same exact prefix IDs and fresh equivalent KV, three times. Use an
excluded warm-up. End after one cycle. Compare variable-length committed output
with the corresponding target trace slice. Replay prefill is setup work, not
runtime overhead; actual draft, verification, reconciliation and cycle costs
are included. It is diagnostic work, not a deployable acceleration bound.

Historical features are acceptance from a canonical k4 probe eight tokens before
the checkpoint and the current context length. Fit a simple bucket rule
(acceptance ≥0.5, context ≥256) on prompts 00–02; validate on 03–05. Unseen
buckets use the fit-global action. Require different hindsight winners and
predicted within-request action variation as well as >5% validation advantage
with bootstrap lower bound >1.0. A scripted 16 target-only cycles followed by
speculation measures actual catch-up, including rollback and draft execution.

For a conservative opportunity estimate charge .05ms decision allowance,
one entire historical probe every 16 cycles and median transition catch-up
amortized over four cycles. These are explicitly **estimates**, not measured
controller gains. The probe's history differs from a future online policy's
history. A passing Q2 result justifies implementing/evaluating the controller;
it does not validate that controller or remove its full execution costs.
With short natural completions, checkpoints can be insufficient; report this
as unresolved, not as absence of adaptation value.

## Statistics and correctness gate

For each prompt/cap/configuration, take the median decode throughput over
repetitions. Pair ratios, take geometric means, and bootstrap distinct prompts
10,000 times (seed 20261107), retaining both caps together. Give categories equal
representation in the fixed suite. Report category ratios, wins/ties/losses at
a symmetric 5% ratio margin, worst regressions and >10% slowdowns. Natural EOG
changes token counts across prompts, never across correct matched configurations.
Repeated timings characterize noise; they are not new workload samples.

No performance analysis accepts pending/failed correctness. Every complete
comparison block requires identical prompt IDs, emitted IDs and stop reason.
Save first mismatches and preserve all failed attempts. On resume rerun a whole
incomplete block, retaining earlier records. Process crashes, malformed logs,
token mismatches and documented external interruptions are invalid reasons;
a slow run alone is valid evidence. Detailed cycle logging is buffered and
must be compared with request-only logging on development data before final
performance claims. `getrusage` peak RSS is lifetime process memory, not VRAM.

## Next stage, deliberately locked

Long development runs can pause at a complete comparison-block or checkpoint-
group boundary using `--budget-seconds`. These pauses are planned scheduling,
not failures or grounds to exclude observations. Keep the source SHA, binary,
model hashes, prompts, order, and settings unchanged on resume. Replay groups
carry content hashes, and prior incomplete attempts remain separate. The owned
Codespace workflow persists measurements outside checkout/lifecycle cleanup,
uploads evidence and confirms shutdown at the end of each invocation.

Held-out report generation now requires the exact frozen measurement grid,
including every repetition/configuration/cap/prompt, with no duplicates. It
requires `selected_fixed` and `development_selection_sha256` in the frozen
configuration. It honors that baseline even when another fixed action performs
better on held-out data. The checked-in synthetic example explicitly declares
its curated single repetition and retains the source-manifest fingerprint;
it does not bypass completeness checks.

Pilot B decision schema 2 also replays the existing dynamic baseline from each
common prefix and requires the predictor to beat both it and the global action
after estimated overhead. For independent models this is pinned confidence
stopping at cap 8. For ngram it is upstream variable-length lookup at cap 8,
which overlaps fixed8 and is labeled accordingly. Older schema-1 replay results
cannot clear the strengthened gate. The active `87c7811` pilot still supplies
valid frozen Pilot A blocks; replay its completed traces under schema 2 before
implementing a controller. The standalone replay command verifies identical
binary/model hashes and execution settings, so an analysis update can consume
older Pilot A traces without mixing measurement implementations:

```bash
.venv/bin/python -m bench.replay --exe build-cpu/tokenomics \
  --target TARGET.gguf --draft DRAFT.gguf \
  --pilot-dir COMPLETED_PILOT_A --out NEW_PILOT_B_DIRECTORY
```

This command does not run held-out evaluation or claim the project is complete.

The 90 held-out prompts and five-repetition protocol are versioned, but their
configuration list stays empty until feasibility selects a contribution.
If Q1 and Q2 pass, implement/calibrate the cost-aware controller, test real
fallback/re-entry, freeze priors/settings and select comparable baselines on
development. Then evaluate target-alone/resident, tuned fixed, confidence,
acceptance and cost-aware, plus no-fallback/probe/init ablations. The executable
and current entry point do not pretend that stage is implemented. If only
request-level selection is supported, make that contribution explicit; if
neither drafting mechanism helps, report the bounded negative result and seek
a technical scope decision. No generic profiler/dashboard pivot is authorized.
