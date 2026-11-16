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

### Calibration context coverage

The runtime confirmed the user's requested model switch to `gpt-6.1-sol`.
Work continues on the same research branch and deadline.

Auditing historical fit priors found nine timing observations but only three
independent prompt/checkpoint states per action. Only one M3 state and two CPU
states produced proposals. Added an offline coverage audit and an optional
frozen `replay_checkpoints` schedule. A denser cap64 fit can sample checkpoints
9/16/32/48 without changing held-out reservations or simultaneous snapshot
storage. The default schedule is preserved. Replay rejects a drafter or
schedule that differs from the fit manifest before launching inference.

The first coverage-audit fixture test exposed missing synthetic labeling in the
audit's call to calibration. It now derives that label from the frozen fit and
includes it in the report; trained and synthetic evidence remain distinguished.

Validation: five calibration/pipeline tests and five replay tests pass, including
four-state suffix calibration. CTest fixture/tooling suites pass in 24.93
seconds. [Retained coverage reports](../results/verification/calibration-coverage-20261116/)
record per-action contexts, proposal availability and immutable source identities.

### Local runner evidence audit

Added complete correctness-gate receipts and an offline verifier for saved
request settings, exact output IDs/stops, cycle accounting and comparison
coverage. A CPU fixture test verifies a passing gate and rejects a changed raw
output. The paired Metal and suffix gates remain hardware-unverified. The Metal
build command now explicitly disables sanitizers to prevent cached diagnostic
flags from contaminating a future timing run.

### Direct backend row diagnostics

Added an optional native matrix-row test and wired it into the Metal runner
before trained inference. The test evaluates actual backend graphs for batched
and independent one-row inputs, comparing finite F32 results bitwise. It covers
F32/F16/Q4_K/Q6_K, widths 64–1536, batches 2/3/5/7/9 and independent/broadcast
weight heads. CPU execution passes; Metal remains unexecuted here.

During expansion to broadcast heads, the first harness version passed the input
head count instead of weight head count to weight quantization. The CPU run
aborted; the quantization count is corrected. This was a diagnostic-harness bug,
not an inference speed result. The corrected grid and sanitizer result are
retained separately. No failed timing is reported as a gain.

Corrected direct-backend checks pass 210 CPU matrix cases and 419,328 bitwise
comparisons in both Release and address/undefined sanitizer builds with leak
detection. [Evidence](../results/verification/backend-rows-local-20261116/verification.json)
records source/binary hashes and the failed harness-attempt hash. The test is an
explicit optional target; it does not add GPU requirements to default CTest.


### Softmax reduction and KV padding audit

A further native Metal audit found that softmax thread count depends on the
number of query rows as well as width and heads. Extended the opt-in generated
host dispatch to use ordinary one-row reduction geometry for short softmax
batches. The first broad text-anchor check correctly refused configuration
because the same expression appears in another operation; narrowed the patch
to the complete unique softmax selector block. Both generated C++ files compile
and 320 host-dispatch cases pass.

Added 48 direct masked-softmax cases (1,053,696 bitwise outputs) to the backend
harness. CPU Release passes them plus the earlier 419,328 matrix comparisons.
The trained M3 gate now includes six KV-padding boundary prefixes, raising its
default context2048 grid to 510 requests. A context512 fixture exercises five
boundaries and therefore480 requests. This targets another possible numerical
source: larger padded attention widths can alter reduction order even when
matmul uses the same kernel. The Metal candidate remains default-OFF and no
Metal numerical or speed claim is made.

### Frozen suffix validation replay

Evaluated the already committed minimum5/cap1 rule on the six archived
**development-validation** prompts, caps64/128; no additional search or tuning.
Reproduced 144 archived ngram1/ngram8 aggregate work checks across both operating
points and timing repetitions. Repetition0 supplies one trajectory per cell.

M3: suffix requires951 cycles, accepts201/220 proposals and wastes19 rows, versus
ngram1's920 cycles,232/273 acceptance and41 wasted rows. Linux: suffix requires
1098 cycles,54/74 acceptance and20 wasted rows, versus ngram1's1050 cycles,
102/142 acceptance and40 wasted rows. Each totals1152 canonical tokens. The
fit-set zero-rejection finding does not persist in validation. Proposal quality
improves at the cost of fewer saved cycles; no speed conclusion follows.
The reserved90-prompt held-out evaluation remains untouched.

### Verification stage instrumentation

Historical `verify_ms` combines target execution and upstream CPU sampling.
A per-prompt least-squares decomposition into fixed cycle cost and per-row cost
on the six M3 fit conditions estimates fixed cost1.47ms (code),1.39ms (QA) and
−0.79ms (summary). This inconsistent fit does not support a suffix speed forecast.

Added default-OFF `TOKENOMICS_VERIFY_STAGE_TIMING`: explicit completed-target
synchronization before the unchanged upstream sampling call, with nested
compute/sampling cycle and request totals. The build marks its identity and
the main Metal comparison explicitly disables it. These diagnostic timings
include host launch/graph work and must not be labeled GPU-kernel measurements.
[Meaning and local M3 commands](verification-timing.md).

The first instrumented full-test invocation overlapped the final executable
relink. Benchmark tests observed an executable permission/fingerprint change,
so that run is invalid for suite validation. After the build completed, the
specific stage-accounting test and the updated runtime suite passed. A fresh
full CTest run is performed only after build completion. Default-OFF Release
already passed all seven suites in39.84 seconds.

Final validation: instrumented CPU build passes all seven suites in38.81
seconds, including1134 runtime cases and explicit stage-accounting checks.
Four separate cross-profile fixture requests preserve prompt IDs, output IDs,
stops and work counters; default output omits the optional timing fields.
[Retained evidence](../results/verification/verification-stages-local-20261116/summary.json)
includes raw synthetic cycle/request records, compiled identities, successful
suite logs and the invalid overlapping-run hash. No trained/Metal execution or
speed gain is claimed.

### Redundant target sampler pass

Identified a removable temperature-zero filter before the target's greedy
argmax. Added default-OFF `TOKENOMICS_GREEDY_NO_TEMP`, with pinned-chain shape
checks and a distinct runtime marker. Model suppression remains earlier in the
chain, and upstream sampling/acceptance still owns verification. Draft samplers
are untouched. The main Metal short-MV runner explicitly disables this option
for both profiles to isolate its kernel experiment.

Actual upstream sampler tests pass384 cases through151936 vocabulary entries,
including ties, infinities, NaNs and repeated maxima. Candidate CPU runtime
passes all eight suites in38.64 seconds (including1134 runtime fixture cases).
A separate330-request cross-profile synthetic gate passes336 comparisons,
exercising3840 drafted/accepted tokens. An initial attempt used real corpus
text with a64-token byte-tokenizer fixture context and stopped at the ordinary
reference with a prompt-too-long error; the retained passing gate explicitly
uses synthetic periodic prompts. No trained cross-profile result is claimed.

Cache-warm CPU component tests include candidate construction and alternate
order over16 trials,32 calls each. Median original/candidate times at151936
entries are244.4µs/158.7µs (paired ratio1.54); at32768 entries52.7µs/35.0µs.
This excludes inference, backend synchronization and model suppression and is
not M3 or end-to-end throughput evidence. Both ordinary and speculative target
sampling can benefit. [Design and M3 gate](greedy-sampler.md).

Greedy-profile address/undefined sanitizer checks with leak detection pass the
sampler equivalence, fixture setup and1134-case runtime suites in286.84 seconds.
The standalone final equivalence binary also passes after adding the optional
component benchmark. Default build configuration is restored with the option
OFF. [Retained evidence](../results/verification/greedy-sampler-local-20261116/summary.json)
includes raw component trials, a complete hashed synthetic cross-profile gate,
source/binary identities and original/candidate compilation settings. The
reserved held-out set remains untouched.

### Native proposal replay on archived target IDs

Added `test_suffix_replay`, an optional helper linked to the actual production
suffix function, plus a source/identity-checking driver. Native C++ reproduces
all24 frozen validation prompt/cap cells across M3 and Linux canonical outputs.
The helper only sees committed prefix IDs when retrieving proposals; the
canonical future is used by the simulated verifier. No weights or target model
execution occurs. This closes the gap between Python proposal-count replay and
the deployed retrieval implementation, while preserving the stated assumption
that alternate target execution paths must still pass a trained exactness gate.

Native replay passes in Release and address/undefined sanitizer builds with
leak detection; their24 output records are byte-identical. An automated test
also confirms that an incorrect expected acceptance count fails. The driver
freezes executable/source identity and refuses mutation during checking.
[Evidence](../results/verification/native-suffix-validation-20261116/summary.json)
includes canonical input IDs, native results, binary identities and hashes.

### Padding-safe verification microbatches

The pinned cache applies each microbatch before deriving its padded KV extent.
For the adapter's contiguous sequence0, a short batch can therefore end its
first chunk exactly at a256-cell boundary, preserving the padded width seen by
ordinary rows. Added this split to the short-MV candidate without discarding
proposals or changing logical work counts. Defaults retain the previous
single-row guard; its generated context copy is byte-identical to the earlier
integration. Prefills/catch-up retain their original behavior.

Host tests check589896 split cases, consuming every row within the same padded
extent as its one-row counterpart. Both generated context translation units
compile. An optional Linux CPU probe links the generated context implementation,
sets the offload flag solely to activate its split condition, and runs actual
synthetic transformer graphs. Periodic and random Q4_K fixtures each pass40
boundary cases and49780 bitwise logit comparisons, with final cache positions
preserved. The first periodic probe incorrectly required most logits to be
nonzero, despite its intentionally sparse output weights; the nondegeneracy
check was corrected to allow sparse valid outputs. No Metal result is claimed.

The audit also identified short-prefill coverage gaps: the short-MV selector
changes matrix kernels for short prefixes or small final prefill chunks, even
when ordinary generation remains one-row. Added seven explicit prefix lengths
3/4/9/10/515/521/522 to coverage version2. The default context2048 gate now has
720 candidate requests; context64/512 fixture grids have450/600. Legacy gate
receipts without a version remain verifiable as version1 under their original
coverage, with a retained-archive regression test. The extra cases can reveal
an unsafe candidate rather than hiding its prefill differences.

Extended the context probe to reject a speculative suffix, retain one row and
verify again, exercising slot reuse and re-entry at boundaries. Each model now
passes99560 bitwise logit comparisons across40 context/rollback cases, in both
Release and sanitizer builds. The589896-case split property test also passes
address/undefined sanitizers with leak detection. Default Release passes all
seven suites in38.50 seconds; five gate tests pass, covering version2 grids and
legacy version1 receipts.
[Retained evidence](../results/verification/metal-kv-split-local-20261116/summary.json).

### Standard hosted Apple Silicon compilation

Added an explicitly selected `metal-research` dispatch to the existing portable
workflow. It is restricted to this research branch and a public repository,
uses the standard `macos-15` runner (not a larger paid runner), and stops after
30 minutes. This uses GitHub Actions, not Codespaces. The job installs pinned
build dependencies, probes `MTLCreateSystemDefaultDevice`, compiles default and
short-MV profiles, runs native fixture suites and direct backend row checks,
and retains only explicit JSON/log files for seven days. A missing GPU is
reported as unavailable; it cannot establish Metal execution correctness.
Neither hosted M1 execution nor synthetic graphs establish local M3 throughput.
See [GitHub runner documentation](https://docs.github.com/en/actions/reference/runners/github-hosted-runners)
and [preflight implementation](../scripts/hosted_metal_preflight.py).
