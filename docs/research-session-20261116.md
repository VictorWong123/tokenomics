# Throughput investigation, 16 November 2026

Session window: 01:50–13:50 UTC (15 November 20:50–16 November 08:50
America/New_York). All changes and experiment source stay on
`research/speculative-throughput-20261116`; the user explicitly prohibited work
on `main`. That instruction supersedes older documentation's main-push route.
Start commit: `2260c1d`.

User update: Codespaces quota is exhausted. Do not start or retry Codespaces.
Use Cloud for correctness and component diagnostics, and prepare a local runner
for the user to execute after the session. Cloud timings do not establish speed
on their machine. The repository is now public; exclude credentials and private
data from committed files, retained artifacts and logs.

## Working hypothesis

The completed CPU controller result is 0.9709 times ordinary throughput and
0.9943 times frozen fixed1. The Metal result is 0.9191 times ordinary throughput.
Microsecond controller overhead cannot explain the inference-stage bottleneck.
Guarded CPU avoids batch-dependent arithmetic, and Metal explicitly serializes
short verification batches. Changing the controller alone cannot make a linear
verification cost profitable when proposals are rejected.

First investigate an opt-in CPU kernel that reuses quantized weight unpacking
across verification rows while preserving each row's existing arithmetic and
reduction order. Unlike the earlier packed profile, it must retain the guarded
ordinary decoder and pass cross-profile equality. Start with Q4_K AVX2, test
bitwise dot-product equality and component timings, then full fixture tests.
No default changes or trained throughput claims follow from a microbenchmark.

Second investigate better cheap proposal selection only after establishing
verification headroom. Longest-suffix matching and match-dependent draft length
are plausible alternatives to fixed three-token lookup. Keep the current ngram
baseline and development/validation split; do not tune on held-out prompts.

## Primary papers inspected

- [Leviathan et al., ICML 2023](https://proceedings.mlr.press/v202/leviathan23a.html):
  the draft cost and parallel verification assumptions matter as much as
  acceptance. Our greedy runtime remains narrower than general speculative
  sampling.
- [Chordiya, 2026 preprint](https://arxiv.org/html/2607.17283v1): consumer-hardware
  measurements identify expensive drafting and effectively serial verification
  as failure modes. This supports measuring the actual backend, not importing
  its reported speedups into this project.
- [Oliaro et al., SuffixDecoding, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/file/b7aea253ab34a773967f1e4cdea9e4fb-Paper-Conference.pdf):
  suffix trees and match-dependent speculation exploit repetition in agentic
  workloads. Our independent requests and cap-eight chain differ from its
  cross-request caches and tree verification. A chain-only prototype would not
  be a reproduction of the full system.
- [Hu et al., SAM Decoding, ACL 2025](https://aclanthology.org/2025.acl-long.595/):
  suffix automatons support longest-suffix retrieval from generated history
  and static corpora. A request-local variant avoids cross-request state but
  cannot assume the paper's gains or corpus coverage.
- [He et al., Thinking Machines, 2025](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/):
  fixed reduction order is essential for batch invariance. This supports sharing
  weight unpacking without changing the original per-row floating-point sums.
- [Zhang et al., 2025 preprint](https://arxiv.org/abs/2505.22179): quantized
  verification can become compute-heavy. Its hierarchical GPU/tree solution
  differs from this CPU chain runtime; the relevant lesson is to measure the
  added quantized verification work.

## Validation and experiment rules

Keep the clean upstream pin, immutable historical results, exact IDs/stops,
charged catch-up, baseline execution settings and current held-out reservation.
Compile and check locally before publishing the research branch. The user has
superseded the earlier Codespaces route: use no more Codespaces execution during
this session. No main push is needed. Final performance measurements belong to
the user's local machine, with both baseline and candidate in the same run.

Record all unsuccessful experiments as well as gains. Compare candidate
speculation with both its ordinary decoder and the unchanged ordinary baseline.
Freeze any tuned settings before new validation measurements. Stop new work
with enough time to retain evidence and shut down remote compute by the deadline.

## Progress

- Inspected existing CPU/Metal evidence, controller design, rejected tiled/cache
  studies and the upstream audit. Remote main matches the local starting commit.
- Created the research branch before edits. Installed development dependencies
  and restored the pinned upstream submodule. Baseline Release build/tests are
  complete: all six suites pass (18.60 seconds).
- Implemented the opt-in Q4 pair kernel and direct bitwise tests. The initial
  component harness omitted `ggml_cpu_init`, producing zero outputs; those
  measurements are invalid. The corrected harness initializes the CPU tables,
  rejects degenerate outputs, and passes 2,048 bitwise dot-product checks.
- Corrected component measurements: two-row matrices improve by median paired
  factors of 1.236 (width4096, outputs4096) and 1.590 (width2048, outputs32768).
  Three-to-nine-row cases improve by 1.134–1.355. Sixteen alternating-order
  repetitions per shape/row count; no model-throughput inference follows.
- Q4 pair Release passes all seven CTest suites (18.35 seconds). A separate
  actual two-executable fixture grid passes all 72 records and its frozen-grid
  audit; changed-protocol analysis is rejected. All seven address/undefined
  sanitizer suites pass with leak detection (395.03 seconds). No remote
  resource had been started at this checkpoint.
- Added the fixed, branch-dispatched `q4-pair` study: fresh 272-comparison trained
  gate, then 72 paired development records. Both profile-specific ordinary
  references and the unchanged guarded decoder are reported. Exact IDs/stops
  and equal work per paired policy are required before speed comparisons.
- Strengthened the paired fixture test to require actual drafted and accepted
  tokens. The first random fixture never drafted; it could check baseline
  equivalence but not exercise the paired verification path. Replaced it with
  the Q4 period-nine fixture. All 72 records now pass with nonzero speculation;
  the negative coverage check is retained separately.
- Added `.inc` files to source fingerprinting, with a mutation regression test,
  so changing the kernel invalidates benchmark resume identity.
- An exploratory offline replay of the three archived fit continuations finds
  that longest-suffix matching (lengths 3–32) alone barely changes the work:
  cap1 saves one rejected proposal; caps2/4/8 have identical cycle, proposal and
  acceptance totals. This is not timing evidence and does not justify replacing
  the drafter yet. Kernel efficiency remains the first experimental direction.
- Recomputed-summary audit rejects an altered summary without rewriting its
  retained bytes. The owned lifecycle suite passes all 24 tests. Release
  formatting and staged-artifact checks pass. First checked stage is ready for
  branch-only publication and explicit `q4-pair` dispatch.
- Published checked source `6d133ef` to the research branch. Attempted run
  [43790995111](https://github.com/VictorWong123/tokenomics/actions/runs/43790995111)
  failed in initialization before any Codespace start request: the legacy
  admission check requires a private repository, while the repository is now
  public. `claimed` remained false; workload and finalizer were skipped. This
  supplies no inference/performance result or observed Shutdown assertion.
  The user then prohibited Codespaces because quota is exhausted. Do not retry
  or change lifecycle credentials/admission to work around that constraint.
- Checked all 35 files published in the first research commit for GitHub/API
  token patterns, private-key headers and credential-bearing URLs: no matches.
  This is a bounded pattern check, not a guarantee about all repository history.

### M3 priority update

The user confirmed **Apple M3 with Metal**. The x86 Q4 pair prototype remains a
separate component experiment; it is not an optimization for their GPU. The
next candidate targets the existing Metal one-row verification split.

- Added opt-in `TOKENOMICS_METAL_SHORT_MV`, default OFF. Generated host dispatch
  copies use existing mat-vec kernels for two-to-nine-row batches and leave
  ordinary one-row and larger-batch dispatch intact. Runtime MMA fusion is
  excluded for short shapes; structural fusion planning stays unchanged.
- Both generated host sources compile on Linux. A host-only program passes 320
  predicate cases across five dtypes, 32 row counts and two head shapes. This
  does not execute Metal shaders or establish whole-graph invariance.
- Added a local M3 runner and cross-profile gate, starting with the original
  `development-code-03` failure case. The broader gate has 330 requests covering
  both caches, fixed caps, scripted fallback, resets, minimal logs and context
  boundaries. CPU fixtures pass the orchestration test and simulated divergence
  test; they are explicitly not GPU arithmetic evidence.
- The runner enables a paired 72-record development screen only after the gate
  passes. It loads only the pinned 0.5B target, accommodating the user's M3
  memory constraints. No Metal timing or exactness result is available here.
- Local instructions: [Metal short-MV experiment](metal-short-mv.md).

### Proposal-quality investigation

- Added a causal offline replay of both archived fit points (M3 Metal 0.5B and
  Linux CPU 3B), using only the three declared fit prompts at 64 tokens. It
  reproduces all 30 upstream fixed-policy aggregate work counts. The archives
  contain empty minimal-log cycle files, so no cycle-by-cycle match is claimed.
- Explored ngram2/3, longest suffix minima 2/3/4/5/6/8/12 and two-occurrence
  consensus. At cap1, minimum-five suffix matching removes all wasted rows on
  both fit points: M3 proposals/acceptances 31/31 versus ngram 37/33; Linux
  23/23 versus 38/33. Fewer wasted rows trade off against fewer saved cycles.
  This is fit-set mechanism selection, not speed or held-out evidence.
- Implemented opt-in `drafter=suffix`, preserving the upstream verifier and
  unchanged `ngram` baseline. A bounded backward scan chooses longest, then
  most-recent matches and reads only committed history. Added independent
  retrieval tests, runtime fallback/rejection checks and calibration binding.
- Release: seven suites pass; 9,216 retrieval cases and 1,134 runtime fixture
  cases. The final Release run passes all seven suites in 32.70 seconds,
  including both 72-record study orchestration tests and suffix calibration.
  Address/undefined sanitizers with leak detection pass suffix, fixture and
  runtime suites in 299.81 seconds (the existing Q4-pair sanitizer build).
  [Retained evidence](../results/verification/suffix-local-20261116/README.md).
- The pinned 0.5B trained-model download failed at the environment proxy with
  HTTP 403. No trained inference was run, and no alternative credential or
  network bypass was attempted. Local model verification remains necessary.
- [Yemets et al., October 2026](https://arxiv.org/abs/2610.31479) evaluate several
  speculative methods on Apple systems and report no gain in their common
  single-request workload. This reinforces measuring the runtime/hardware
  combination; it is not a proof of impossibility for our M3 setup. Its
  supplementary cases include positive M3 Max ngram and M3 Pro BF16 tree/block
  results, emphasizing the scope of the negative common benchmark.
- [Dovetail, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.879/) uses GPU
  drafting and CPU verification for heterogeneous systems with constrained
  GPUs. Its device/model balance differs from the M3's shared-memory 0.5B
  target; adding a second model is not justified by its published gains alone.
- Auditing GPU sampling revealed that the pinned Metal argmax reductions use
  maximum-index tie selection, unlike the host greedy sampler's first maximum.
  Backend sampling also retains sampled-logit outputs. Enabling it alone would
  neither establish exact ties nor eliminate all host vocabulary work, so it
  has not been enabled.

- Added the guarded-M3 suffix study runner: a 330-request correctness gate before
  a 72-record paired screen against ordinary, ngram1 and ngram8. The synthetic
  gate, full screen, immutable-record audit and tamper rejection pass on CPU.
