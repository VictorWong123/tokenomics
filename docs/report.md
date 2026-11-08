# Tokenomics: portable speculative-decoding runtime engineering

Technical report, implementation checkpoint, 8 November 2026.
The [four-page engineering PDF](../results/verification/engineering-report-20261108/tokenomics-engineering-report.pdf)
provides a concise, verified checkpoint; this document retains the fuller technical history.
This is an in-progress engineering result, not an acceleration or research
novelty claim. The full feasibility and held-out evaluation remain incomplete.

## Objective and current conclusion

Speculative decoding tries to reduce autoregressive generation time by asking
a smaller model for several likely tokens, then verifying them in one larger
target-model pass. Its benefit depends on both acceptance and execution cost.
Drafting can be too expensive, batched target verification can save too little,
and synchronization can erase an apparent benefit. A controller is useful only
if speculation first helps and available observations predict useful changes
in the preferred action.

The user explicitly chose experimental controller completion despite the
negative feasibility results. The C++ runtime now includes a calibrated
six-action EWMA controller, fully charged transition scoring, hysteresis,
rotating probes, target-only fallback and both ablations. Its
[design and implementation contract](controller-design.md) was committed before
implementation; no trained performance priors or gains were fabricated.
The two full feasibility questions remain unresolved. Trained correctness
passes in Codespaces, and the completed frozen-action confirmation gives no
support for a practical gain on the guarded CPU point: cap1/target throughput
is 0.9846, with 95% interval [0.9721, 1.0019]. The controller has now completed its separate bounded development study:
1.2048× cap8 ngram, 0.9943× the frozen fixed1 baseline and 0.9709× ordinary
decoding. Its 1.67 µs/cycle decision/update cost is small; native verification
dominates. The [dedicated result](controller-results.md) reports uncertainty,
work counts, overhead and ablations. The separate packed CPU profile passes its
within-profile trained gate, but the paired comparison fails cross-profile greedy
equality at output index36. No CPU-profile speedup is accepted; guarded remains
the default. A subsequent original-history diagnosis reproduces both full
outputs and shows reversed top-two ordinary target logits at that index. Its
workflow report failed after inference; the original archive and actual Shutdown
are independently verified with that failure recorded. The older partial development pilot remains exploratory and is not
combined with the confirmation.

| Environment | Actual execution | Feasibility evidence |
|---|---|---|
| Codex Cloud, Linux CPU | Release and sanitizer builds; synthetic-transformer correctness; resumable experiment and analysis | No trained-model evidence |
| GitHub Codespaces, two CPUs | 272 trained-model comparisons; two timing screens; 66 original-KV replay checks; owned Actions lifecycle and confirmed shutdown | Preliminary slowdown; full Q1/Q2 unresolved |
| User's Mac, Metal | Commands and build configuration supplied | No attached execution route; hardware/RAM uninspected |

The development host exposes an AMD EPYC 9V74 with five logical CPUs, but its
cgroup quota is four cores. Memory is limited to 16 GiB with no swap or GPU.
These observations informed the candidate screen; host-reported total RAM or
CPU count alone would overstate the allocation. Shared-cloud timing would apply
to that allocation, not to an M3 Mac or all computers.

## Implementation and ownership

llama.cpp is an unmodified submodule pinned to
`fd27b2034905f8dadd8bd6d2cc44595f4929d391`. CMake rejects a different revision.
Upstream supplies loading, tokenization, tensor execution, draft generation,
sampling and greedy speculative verification. In particular, Tokenomics calls
`common_sampler_sample_and_accept_n` rather than introducing another acceptance
algorithm. An empty draft uses the same upstream greedy sampler for ordinary
target decoding.

Our implementation owns the C++ request lifecycle, policy interface, draft-cap
budgeting, cache reconciliation, fallback/re-entry, measurements and structured
results. It adds fixed caps, the pinned upstream confidence stop and the
project-proposed acceptance heuristic. A scripted action policy is diagnostic
only. Policy code has no backend-specific calls; the adapter uses supported
llama.cpp/ggml APIs. CPU is the default; optional Metal/CUDA builds use the same
policy code. GPU-layer requests fail if no local GPU backend exists, and result
metadata retains available devices and upstream offload messages.

The central state invariant separates a cached committed prefix P from one
pending token L. At a cycle boundary P+L is the exact prompt plus generated
sequence. The target evaluates L followed by proposed drafts, then upstream
returns the matching prefix plus a correction or bonus. The adapter commits
only through the first target EOG or output limit, moves all but the final
emitted token into P, and removes target/draft cache entries after the new P.
The target cache must equal P's length; the draft cache may be a shorter valid
prefix. All positions and removal return values are checked.

The retained-cache path avoids discarding valid draft computation. Before the
next speculative cycle it decodes only missing committed tokens into the draft.
This supports lazy synchronization after target-only execution and handles the
less obvious full-acceptance case: upstream may not have evaluated its final
proposal, so accepting every proposal does not guarantee equal cache lengths.
An upstream-style control discards proposal KV and reprocesses the entire
target verification batch on the draft before trimming. Comparing these paths
can isolate saved draft work, including rejected suffixes. This control
reproduces the supported example's behavior; it is not an execution of the
stock executable. The optional unmodified executable now builds and runs as a
separate diagnostic: on a synthetic cap-three case it reports eight generated
tokens. Its stopping/counting behavior prevents treating its aggregate rate
as an exact-budget baseline. Trained stock performance remains unmeasured.

Only full-attention transformers with arbitrary partial sequence removal are
accepted. Recurrent/hybrid/checkpoint-only contexts, stochastic sampling and
context shifting are outside the adapter's verified contract. The logical
context limit is enforced explicitly because upstream can round its allocation
upward. Vocabulary compatibility is checked more strictly than upstream's
permissive check: size, every token's text/attributes and EOG status must match.

## Correctness evidence

The portable suite uses tiny deterministic/random F32 transformers generated
locally as GGUF files. These are actual transformer forward passes but are not
trained language models. Controlled logits make full acceptance and EOG
reproducible; test-only proposal corruption forces zero and partial acceptance
without replacing upstream verification. Random weights exercise less uniform
token sequences. Tests compare emitted IDs, not rendered text or random seeds.

The current runtime suite passes 906 differential cases, covering zero/partial/
full acceptance, correction/bonus handling, EOG, changing draft caps, rollback,
fallback, draft catch-up, re-entry, repeated-request reset, empty/short requests,
output truncation and logical context boundaries. Unit tests separately cover
commit counting and acceptance-policy bookkeeping. Address/undefined-behavior
sanitizer testing with leak detection has passed on Linux.

The suite now also exercises Q4_0 and Q4_K weights, large-prefill/small-decode
kernel transitions, exact first-row top-two logit equality across draft caps,
and 30 additional minimal-logging/checkpoint comparisons. Current guarded and
packed Cloud Release and ASan/UBSan/leak checks pass all six CTest suites,
including the independent controller and strict JSON tests. See
`results/verification/controller-local-20261108` and
`results/verification/controller-packed-local-20261108`. The earlier four-suite
and 24-lifecycle-test checkpoint remains in `results/verification/cloud-20261108`.

Additional C++ work provides bounded, within-process target KV snapshots for
common-prefix replay. Actions restore the original target history rather than
rebuilding it with different prefill batching; their draft state is prepared
once per prefix and restored identically. This also removes repeated full
prefix processing for every action. Capture/restore/setup costs remain logged
as diagnostics, and those requests are prohibited from performance summaries.
Fixture checks cover 12 groups and 264 canonical continuations, plus exact
first-row logit equality after restoring long-prefix quantized state.
See `results/verification/kv-checkpoints-20261108`. Trained-model validation of
this extension remains separate from the earlier trained gate. The larger-target
screen below now also validates 66 canonical actions from three trained-model
checkpoints. A period-nine fixture additionally forces ngram proposals at every
cap and tests corruption, fallback/re-entry and reset without a draft model;
Release and extended sanitizer checks pass with leak detection.

The K-quant suite now also uses width-512 random weights, widening the
attention/output projections and cache state (the earlier FFN-down projection
already exercised multiple quantization blocks). The expanded default passes all four Release and
address/undefined-sanitized suites with leak detection. A separate forced-all-
tiled configuration passes the fixture checks but is not adopted: on a tiny
perfect-copy diagnostic its target-only cost grows from 30.20 to 158.74 ms,
while cap8 stays about 26 ms in both profiles. This prevents claiming a gain
against a deliberately slowed baseline. The source, raw controlled measurements,
wide logit probes and rejected initial harness are retained in
`results/verification/kernel-study-20261108`. They establish no trained-model,
Codespace or Mac performance result.

An experimental `TOKENOMICS_CPU_PACKED_VERIFY` profile instead keeps native
packed GEMV and its one-row activation quantizer for each of up to nine
verification rows. Larger prefills retain the upstream matrix path; unsupported
packing types retain the dot-kernel guard. Four checked substitutions are applied
to a generated source copy, with no submodule edit or verifier change. This
CPU-only profile is off by default and mutually exclusive with the existing
batch-invariant profile. It passes all 468 differential cases, Release suites
and address/undefined-sanitizer suites with leak detection. Its controlled
fixture improves ordinary decode as well as verification, but trained-model
correctness, memory and throughput must be checked separately. The fixed
`kernel` workflow phase performs that gate and an independent development
screen; it never clears full feasibility by itself.

A separate 128 KiB weight-chunk prototype also preserves one-row quantization
and GEMV arithmetic, and passes the 468-case strict fixture suite. It does not
improve the controlled verification endpoint, so it is rejected. Per-process
measurements and unexplained ordinary-row timing variation are retained in
`results/verification/cache-study-20261108`; this is no trained-model result.

Codespace run 42420516268 passed all 272 comparisons on trained Qwen2.5
0.5B/1.5B Q4_K_M weights, following a documented batch-dependent arithmetic
repair. Repacking, llamafile and BLAS are disabled in this CPU profile; tiled
kernels are kept for larger prefills but excluded from verification widths up
to nine rows. These are correctness constraints, not assumed optimizations.
The pinned submodule remains unchanged; CMake builds a guarded copy of its
tiled selector. See `docs/divergence-investigation.md` for failures and resolution.

## Preliminary trained-model measurements

A reproducible terminal demo can show an actual retained trained trace without
model download: `.venv/bin/python scripts/demo.py --recorded`. It verifies the
original evidence hashes and ordinary target IDs before displaying scripted
caps, acceptance, committed tokens and lazy draft catch-up. The recorded output
is `results/verification/codespace-gate-20261108/terminal-demo.txt`; its scripted
choices are correctness exercises, not cost-aware decisions.

Run 42452778183 used two logical CPUs on an AMD EPYC 7763 Codespace with
approximately 7.8 GiB physical RAM and no swap. It tested three long development
prompts, one per category, at cap 64 with one repetition. All nine comparisons
passed exact prompt/output IDs and stopping checks. The archive was downloaded
and all 61 members verified against its execution manifest; shutdown was
confirmed. Source was `81c44cc8e731c89cc4fc357f295f70f111182b64`.

| Configuration | QA decode tokens/s | Code decode tokens/s | Summary decode tokens/s | Geometric mean ratio to target |
|---|---:|---:|---:|---:|
| Target-only | 14.84 | 14.96 | 13.29 | 1.000 |
| Fixed cap 4 | 6.19 | 7.70 | 7.77 | 0.501 |
| Confidence cap 8 | 7.22 | 8.43 | 7.33 | 0.533 |

This screen provides no repeated-run confidence interval, Q1 decision or Q2
prediction result. It cannot establish that every configuration loses. On code
and summary requests, acceptance reached roughly 67–75%, yet verification alone
still exceeded target-only generation time. Drafting added approximately
2.4–3.2 seconds per 64-token request. Decisions totaled about 0.01–0.02 ms per
request, so their overhead is not the observed bottleneck. Peak lifetime RSS
reached 1.22 GiB for target-only and 1.78 GiB with speculation. All observations,
including the unfavorable ones, remain in the timing artifact.

The original linear pilot estimator reported 54.95 hours. Treating fixed
prefill/load/warm-up costs separately gives about 19.72 hours; neither is a
duration guarantee because other configurations and EOS are unmeasured.
The subsequent pilot uses a bounded invocation and preserves complete blocks
for unchanged-source resume. No full-protocol completion is inferred from a
timing screen. Compact raw records and hashes are retained under
`results/verification/codespace-timing-20261108`.

The second independent-model screen,
[42461309986](https://github.com/VictorWong123/tokenomics/actions/runs/42461309986),
uses the 0.5B draft and 3B Q4_K_M target at source `1f945147`. The same three
development prompts, cap 64 and one repetition compare the complete fixed-cap
sweep and pinned confidence stop. All 21 ID/stop checks pass; all 173 archive
members were independently hash-verified. Geometric mean ratios to target-only
are fixed1 **0.768**, fixed2 **0.723**, fixed4 **0.587**, fixed6 **0.476**,
fixed8 **0.439**, confidence **0.682**. Every observed drafting cell loses.
Peak RSS is 2.75 GiB, below the memory screen; this is not a repeated Q1 result.
The separately extrapolated full Pilot A duration is 29.67 hours; unmeasured
prompts/configurations, EOS and shared-CPU conditions can change that estimate.

The same run validates the new original-KV snapshot path on trained weights:
three source captures and 66 canonical action checks pass, including dynamic
stopping and the historical signal. All checkpoint file hashes, token prefixes,
continuations and cache/commit accounting were independently rechecked. Maximum
snapshot payload is 83,503,856 bytes. Setup/capture/restore remain diagnostic
costs, excluded from action scores. Three fit prompts cannot support the Q2
fit/validation procedure; its decision explicitly says insufficient checkpoints.
Hosted finalize confirms actual Shutdown. Full raw evidence, reproducible
analysis and provenance are retained in `results/verification/codespace-3b-20261108`.

The Python suite checks deterministic balanced ordering, first-divergence
reporting, fit/validation separation, prompt-clustered statistics, atomically
saved artifacts, identical-command resume and configuration-change rejection.
A deliberately incorrect fake engine demonstrates that a token mismatch cannot
produce a complete performance block. The end-to-end fixture workflow has run
168 request comparisons plus common-prefix replays and real scripted catch-up.
Fixture timing is retained only to exercise artifact generation; it cannot
answer whether useful language-model speculation helps.

The cheaper ngram screen in run 42461982414 attempt 2 uses the identical 3B
target hash, binary and execution settings, without a draft model. All 18
comparisons and 66 snapshot action checks pass. All 134 artifact members were
hash-verified and actual Shutdown confirmed. Geometric mean ratios to the
within-run ordinary reference are fixed1 **1.0113**, fixed2 **0.9429**, fixed4
**0.9205**, fixed6 **0.8402**, fixed8 **0.7821**. The shortest cap makes actual
proposals, accepting 5/6, 14/14 and 14/18 across the three prompts. Peak RSS is
2.18 GiB. Its 1.1% aggregate estimate is below the 5% practical margin, with no
repeat uncertainty estimate; larger caps lose. The full ngram Pilot A duration
extrapolation is 12.25 CPU hours.

The frozen targeted confirmation completed on nine other development prompts,
caps 128/512 and three repetitions. It selected fixed1 only from the separate
fit screen and retained that choice through validation. All 54 blocks/162
comparisons pass exact IDs and stops.

| Configuration / ordinary target | Geometric mean throughput ratio | Prompt-clustered 95% interval |
|---|---:|---:|
| Frozen cap1 | 0.9846 | [0.9721, 1.0019] |
| Cap8 comparator | 0.7727 | [0.7395, 0.8063] |

Cap1 has one win, sixteen ties and one loss across the eighteen prompt/cap cells
under the frozen symmetric 5% margin. Cap8 loses every cell; sixteen are more
than 10% slower in time per token. The selected action fails the 5% practical
gain criterion. The three categories are equally represented; repetitions are
collapsed before prompt-clustered resampling.

All 927 archive members, complete-block hashes, output/cycle/cache checks and
regenerated ratio/interval statistics pass independent hosted verification.
The retained controller report confirms actual Shutdown. Peak process RSS is
2.18 GiB. Complete raw bytes and the verification receipt are in
`results/verification/codespace-confirmation-20261108`.
This targeted result does not complete full Pilot A/B, predictive scheduling,
a controller comparison or held-out evaluation. The separate packed CPU profile
is a new operating point, with its own correctness/memory/performance checks.

## Experiment design and timing

Candidate selection begins with Qwen2.5 0.5B/1.5B Q4_K_M as an initial pair,
then 0.5B/3B if available RAM and measured peak RSS permit. Size alone does not
establish a useful cost gap. If both independent drafters fail, the bounded
investigation includes upstream ngram-simple lookup drafting before a scope
decision. Hashes and model repository revisions are resolved and verified at
download; weights remain outside Git. Exact locks must be copied between
machines for cross-platform comparisons.

Pilot A uses 18 development prompts, six each for Q&A, C++ code and summaries,
two output caps and three repeated balanced blocks. It compares target-alone,
resident-draft target-only, upstream-style cache replay, five fixed caps,
confidence stopping and acceptance scheduling. A small measured timing batch
precedes the long run and produces a duration estimate. The resident control
warms both weight sets; merely mapping a draft would understate physical RSS.
Each request resets KV and policy state. Target weights and settings stay
identical within each comparison.

Pilot B uses common prefixes from ordinary target traces. Every action is
replayed from equivalent prefix IDs/KV, one complete cycle at a time, with
three repetitions. A historical acceptance signal and context length feed a
small bucket rule fitted on different development prompts from validation.
The analysis retains actual action cost and separately measures target-only
to speculation catch-up. Conservative decision/probe/transition allowances
are estimates for the feasibility decision, never claimed as observed
controller performance. Hindsight action winners indicate an opportunity,
not an achievable bound.

Monotonic timing separates weight/context loading, request setup, target/draft
prefill, first completed commit block and generation. Prefill excludes the
pending final prompt token in every path. Generation includes drafting, target
verification, decisions, cache work, synchronization and fallback. Logit reads
synchronize upstream target work; prefill/drafting explicitly synchronize
completed work. Warm request time also includes text conversion and teardown.
Whole-process wall time retains loading, excluded warm-up, serialization and
other initialization. First-token timing is a commit-block upper bound, not a
streaming measurement. Stage overlap and backend-specific profiling still need
verification on actual Metal hardware.

RSS is the process lifetime peak from `getrusage`, including load and warm-up.
It is not per-request allocation, GPU VRAM or energy. No reduction in memory,
energy or dollar cost follows merely from lower generation time. Detailed cycle
records are buffered. Minimal mode avoids retaining cycle records while still
collecting timers. The completed controller study separately measures detailed
versus minimal logging on two development prompts; its interval is consistent
with a small effect and supplies no practical aggregate gain.

## Reproducibility and next decision

The repository includes a versioned original CC0 workload: 40 development and
90 held-out prompts with embedded summary documents. These synthetic tasks
have known structural correlations and are not an external quality benchmark.
The final protocol uses five repetitions, exact token-ID correctness gates,
development-selected fixed settings and an appropriate dynamic baseline.
Paired median-throughput ratios are aggregated geometrically and bootstrapped
by prompt with both caps kept together. Repeats are not independent prompts.
Regressions and failed attempts remain visible.

Completed comparison blocks are atomic and resumable; incompatible binary,
model or configuration changes require a new run directory. Partial requests
are saved before interruption. A failed block is rerun as a whole with its
earlier attempt retained. Model files, environments, builds and local full runs
are ignored before pushing. The README provides one build/check command and
one feasibility entry script for CPU or Metal, plus independent analysis and
result-transfer commands.

No held-out controller evaluation has occurred. The experimental C++ controller
passes Codex Cloud Release and ASan/UBSan/leak checks, including 906 differential
cases. Its frozen calibration/measurement pipeline is tested with generated
transformers; model calibration requires an actual catch-up episode and the
ngram calibration has no draft-model synchronization charge.

Trained controller study **42492089188**, source `e98c0fa84a2c2ff52df198f19eae6ce72d51c83c`,
started only after those local checks. Both trained correctness mechanisms pass
28 records. Frozen development priors select fixed1 from complete requests.
All 252 bounded validation records and 16 separate logging records pass exact
IDs/stops/cache accounting. Both attempts have confirmed actual Shutdown;
all 1587 payload files and the execution manifest pass independent verification.
The resumed raw prefix, fit and prior remain byte-identical.

Cost-aware/target throughput is 0.9709 [0.9607,0.9810]; cost-aware/fixed1 is
0.9943 [0.9853,1.0023]; cost-aware/cap8 is 1.2048 [1.1271,1.2868]. The bounded
grid uses six distinct development prompts, caps64/128 and three repetitions;
these prompt-bootstrap intervals do not constitute a full held-out result.
Neither ablation shows a practical aggregate benefit. Policy work is 0.00114%
of generation and the measured prediction error is 22.7%. Fallback supplies
47.3% of tokens. A separate native CPU-profile experiment addresses the measured
verification bottleneck, with an equally optimized ordinary baseline.
See [the controller result](controller-results.md) for full tables and figures.

Pinned-source findings and reference links are in `docs/upstream.md`. Detailed
commands, blockers and current résumé wording are in `docs/handoff.md`.
