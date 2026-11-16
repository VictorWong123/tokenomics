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

### Arithmetic-preserving dispatch locality prototype

Research review confirmed upstream PR29869 is already in our pin. It shares
weight dequantization using few-row MMA, but those reductions differ from the
ordinary one-row reference. Added a separate, default-OFF
`TOKENOMICS_METAL_INTERLEAVE` prototype requiring short-MV and embedded shaders.
For dense Q4_K/Q6_K projections, it swaps the two host grid extents and adds two
shader wrappers that swap them back before calling the unchanged upstream
mat-vec implementations. This may reuse weight tiles in cache between adjacent
activation rows. GPU scheduling and cache benefit are not guaranteed.

Generated common/ops/device C++ and both context copies compile on Linux.
All320 host dispatch cases and589896 padding split cases pass; the original
shader body is preserved except for removed internal includes. Default CPU
passes7/7 suites in38.20 seconds. The trained-model gate requires an explicit
interleaved candidate marker; the ordinary short-MV study forces the extra flag
OFF. Hosted preflight now also compiles this third profile and separately probes
the offline shader compiler. No Metal execution or M3 speedup is claimed.
[Host evidence](../results/verification/metal-interleave-host-20261116/summary.json)
and [research rationale](metal-research-directions-20261116.md).

The first hosted dispatch failed during action resolution because two action
commit pins were assigned to the wrong action names; corrected before retry.
[Run43806416345](https://github.com/VictorWong123/tokenomics/actions/runs/43806416345)
then compiled the native default profile, passed its fixture suites and CPU row
checks, and compiled36 embedded Metal libraries. Its GPU is an **Apple
Paravirtual device**, advertising neither simdgroup reduction nor matrix
support. Direct GPU MUL_MAT execution aborted as unsupported before reaching
the candidate. This is a compute capability limit, not a correctness pass.

Updated preflight to probe the same Apple7/Metal3 reduction families as the
pinned backend, explicitly distinguish compilation from graph execution, and
compile embedded libraries without executing graphs on limited devices. The
reference context guard does not apply to raw backend graphs; the strict row
invariance execution test belongs to candidate profiles. A missing required GPU
feature is reported as unavailable. Artifact bytes could not be downloaded into
this execution workspace through its proxy; the public failure excerpt and
fail-fast stage ordering are retained without claiming absent successful logs.
[Retained negative evidence](../results/verification/hosted-metal-20261116/summary.json).

### Request-local retrieval ceiling on frozen fit trajectories

Added a future-aware oracle diagnostic to distinguish retrieval limitations
from backend verification cost. It only copies already observed tokens but uses
the canonical future to choose the occurrence/length and computes the minimum
cycle path. This is deliberately non-deployable: no rejection or lookup cost,
no cross-request corpus, no latency prediction. All15 archived ngram policy
counts per platform (30 total) are revalidated before comparison. The implementation
matches an independent breadth-first enumerator on2480 short trajectories.

For192 fit tokens per platform, the minimum-five-token, cap1 suffix already
matches the oracle's cycles:161 on M3 and169 on Linux. At cap8 the oracle and
causal suffix also share137/155 cycles, while the oracle removes5/38 wasted
rows. Relaxing matches to one token gives an optimistic147/145 cycles at cap1
or115/111 at cap8, with perfect future-aware filtering. This limits the fit-set
benefit available from merely improving occurrence selection; it does not rule
out learned proposals, cross-request retrieval or other workloads. The90
reserved held-out prompts remain untouched.
[Method and table](retrieval-envelope.md),
[all90 oracle traces](../results/verification/retrieval-envelope-20261116/envelope.json).

Added an optional `--screen retrieval` to the local short-MV runner. Its new
frozen14-configuration grid includes suffix caps1/4/8 on both backends alongside
ordinary decoding and ngram caps1/4/8. Both ngram and suffix trained exactness
gates must pass before its126 paired fit records. The original default grid
and its summary state remain compatible. Reports compare against ordinary and
ngram8 baselines, require the same logical work across kernels, and identify
fit-set cap exploration. Seven fixture-backed gate/analysis tests pass in9.56
seconds, including both grids, tampered outputs and failed-gate retention. No
M3 timings were generated by these CPU fixture tests.

### Inspecting actual quantized tensor coverage

The target's896-wide hidden state is not divisible by256, so its `Q4_K_M`
filename cannot establish that every large projection uses a K-quant kernel.
Added a bounded public-model header inspector: it reads at most16MiB from the
locked revision, parses GGUF metadata and the tensor table, and saves no weight
bytes or tokenizer vocabulary. Reports explicitly say that a header-only read
does not verify the full model SHA256. Range/full-size headers must match the
lock entry. Three parser tests pass, including every truncated synthetic-header
prefix; it also parses the existing21-tensor generated Q4_K fixture correctly.

An explicitly selected `model-metadata` workflow dispatch uses a standard
Ubuntu runner with a five-minute timeout and the same public-research-branch
restriction. It runs no benchmark and uploads only header-derived JSON. Its
purpose is to establish which tensor types and shapes the M3 candidate must
cover, rather than assuming coverage from the model filename.

[Run43807666479](https://github.com/VictorWong123/tokenomics/actions/runs/43807666479)
provides actual Apple Silicon compilation for both guarded and short-MV profiles:
native executables, offline Metal3.1 mat-vec/softmax shaders, CTest suites, CPU
row comparisons and embedded-library initialization all pass. The orchestration
then failed because the capability probe still omitted a newly required key;
the capability-source replacement had not applied. Fixed the actual Swift
source and added an immediate schema check. A profile selector permits checking
only the not-yet-built interleaving prototype without repeating passed builds.
The original failed state and18 successful-stage hashes remain in
[two-profile evidence](../results/verification/hosted-metal-20261116/two-profile-partial.json).
No GPU graph or M3 throughput result follows from these compilation passes.

The first metadata dispatch stopped before network access because its model
lock is deliberately local/ignored. Added an explicit tracked JSON containing
only the two public Qwen revisions, filenames, licenses, byte counts and hashes
already used by the session. Header inspection now reads that public pin file,
so a clean checkout does not depend on a developer's ignored model directory.

The [header dispatch](https://github.com/VictorWong123/tokenomics/actions/runs/43809348043)
succeeded:291 tensors, including133 Q5_0 and13 Q8_0; output.weight is Q8_0,
896×151936, encoded in144643072 bytes. The first query projection is Q5_0;
FFN down-projection width is4864. The initial Q4_K/Q6_K-only interleaving
prototype therefore misses the large output matrix. Expanded it to Q5_0 and
Q8_0, retaining the original shared-memory parameter and Q8_0's different host
grid divisor. Four wrappers still call unchanged upstream reductions.

The direct backend harness now includes both actual types and4864-wide inputs:
450 matrix cases/898560 bitwise values plus48 masked-softmax cases/1053696
values pass on CPU in Release and address/undefined sanitizers with leak
checking. The existing sanitizer profile has CPU Q4-pair ON; that fact is
retained rather than presenting it as the default build. Host compilation and
448 dispatch cases pass, as do589896 cache-split cases. The original shader body
is preserved. These are host/CPU checks; expanded Metal shader compilation and
full local M3 exactness remain separate requirements.
[Expanded evidence](../results/verification/metal-interleave-host-20261116/expanded-summary.json).

The initial two-type interleaving prototype passed its native Apple hosted job:
C++/Mach-O linkage, offline Metal shader compilation, native fixture suites,
CPU row arithmetic and embedded-library initialization all pass. The job
explicitly reports GPU graph execution unavailable due to missing simdgroup
reduction. This result belongs to source91f9a56 and does not establish the
subsequent four-type expansion's native compilation.
[Initial native receipt](../results/verification/hosted-metal-20261116/interleaved-initial-native.json).

### Bounded real-model CPU smoke on hosted compute

Added a30-minute standard public Ubuntu job to exercise the pinned learned
weights beyond synthetic fixtures. It builds guarded and greedy-only sampler
binaries, reusing compiled dependencies while preserving the first executable,
then verifies the full downloaded model SHA256. The smoke compares160 requests
across ordinary decoding, ngram/suffix caps1/4/8 and scripted fallback, with
four development prompts, tiny budgets and two prefix boundaries. It audits
IDs/stops, cycle accounting, drafter participation and equal logical work across
sampler profiles. It is not a performance experiment or a Metal execution test.
Only explicit correctness/configuration files are uploaded; weights and binaries
are excluded. The actual learned-model result remains pending.

Before dispatch, its orchestration passes160 synthetic comparisons and a
negative wrong-sampler-profile test in1.72 seconds. The test explicitly labels
those outputs synthetic. [Smoke contract](trained-cpu-smoke.md).

The [expanded native job](https://github.com/VictorWong123/tokenomics/actions/runs/43810371502)
passes all10 stages for the four-type candidate, including Apple C++ linkage,
Metal3.1 shader compilation, native fixture suites and CPU row comparisons.
GPU graph execution remains explicitly unavailable on the paravirtual device.
[Expanded native receipt](../results/verification/hosted-metal-20261116/interleaved-expanded-native.json).

Added `--interleave` to the existing local study runner so the new prototype is
concrete and runnable: it builds the guarded reference and interleaved candidate,
checks direct GPU row arithmetic, runs the appropriate trained exactness gates,
and only then times a distinct frozen grid. `--screen retrieval` includes suffix
caps1/4/8 and strong ordinary/ngram8 controls. The report labels the combined
short-MV/interleaving candidate rather than claiming an isolated cache effect.
Eight gate/analysis tests pass in11.42 seconds, covering original, retrieval and
interleaved retrieval fixture grids plus tampered-output rejection. Those tests
remain explicitly CPU orchestration evidence.

### Learned-model CPU result

[The pinned learned-model CPU job](https://github.com/VictorWong123/tokenomics/actions/runs/43811081794)
passes all160 output comparisons and equal-work checks across sampler profiles,
with accepted proposals for both ngram and suffix drafters. It verifies the full
model SHA256 before execution. Across348 canonical smoke tokens, suffix4 uses
296 cycles/353 verified rows versus ngram8's284/422. These totals include tiny
budgets and artificial prefix boundaries; they are correctness/work evidence,
not a workload latency estimate or an M3 result. The printed aggregate summary
is retained; complete raw files remain in the Actions artifact.
[Retained summary](../results/verification/hosted-metal-20261116/trained-cpu-summary.json).

### Restrict short-MV to verification graphs

The previous shape-only selector could change short-prefill arithmetic before
speculation began. The owned context now marks only causal, offloaded,
single-sequence graphs with2–9 tokens and outputs for every row. It uses a
reserved high tensor flag bit on MUL_MAT/SOFT_MAX nodes, before scheduling.
Pinned allocator/Metal code tests named low bits; backend tensor copies retain
flags. A static assertion excludes all named flag bits. Graph reuse checks
output count, so verification and one-output prefill shapes remain distinct.
No global/thread-local execution mode or additional synchronization is added.

Only marked operations select short-MV/interleaving and ordinary-row softmax
geometry. Unmarked prefills/catch-up retain original selectors, including
short tails of long prompts. Host tests check both tagged verification and
untagged upstream decisions while preserving original flag bits. The default
single-row context copy remains byte-identical to its earlier integration.

The actual generated context executes68 prefix/rollback cases per synthetic
model, including all seven short-prefill shapes. A scheduler metadata callback
observes2772 tagged and13335 untagged nodes with no scope leaking into prefill or
ordinary graphs. Periodic and random Q4_K fixtures each pass169252 bitwise logit
comparisons in Release and address/undefined sanitizers with leak checking.
This establishes scope propagation on CPU; full M3 graph exactness remains a
separate gate. [Scope evidence](../results/verification/metal-graph-scope-local-20261116/summary.json).

Default Release remains7/7 suites passed in41.60 seconds. The direct backend
harness also adds the target's14-query/2-KV-head broadcast ratio at five
attention widths. All500 matrix cases/1364480 values and48 masked-softmax
cases/1053696 values pass bitwise on CPU in Release and sanitizers. Native
compilation of the new graph-scoped candidate is the next separate check.

The [graph-scoped native job](https://github.com/VictorWong123/tokenomics/actions/runs/43814656639)
passes all10 stages, including Apple C++/shader compilation, fixture suites and
the expanded tagged CPU backend harness. The paravirtual device still cannot
execute required GPU graphs.
[Retained receipt](../results/verification/hosted-metal-20261116/interleaved-graph-scope-native.json).

### Inspecting compiler output

Added a separate five-minute standard macOS26 compiler-IR diagnostic. It
source-generates the exact owned shader through the same CMake helpers,
requests LLVM-style text, and compares original/wrapper functions after grid
permutation and textual normalization. Arithmetic, FP flags and addresses
remain significant; metadata is deliberately ignored, so this is not a formal
or hardware correctness proof. Three unit tests pass, including negative
arithmetic/flag/address/coordinate changes and missing functions. Source-only
CMake generation also passes locally without building or linking a backend.
The newer runner gets a device-only SIMD capability probe; no paid larger
runner or Codespaces operation is involved. [Diagnostic contract](metal-ir-diagnostic.md).

The first IR job succeeds in emitting266 functions. All four wrapper pairs show
an explicit(1,0,2) `shufflevector` before a helper call, which the initial
extract-only text normalizer correctly reports as different/incomplete. Added
strict cancellation of exactly that observed inverse shuffle and rejection of
any unmodeled whole-grid use; direct callee names remain part of comparison.
Four unit tests pass, including wrong shuffle masks, raw-grid forwarding and
different helper callees. The original report remains retained rather than
being relabeled as a match.

The standard macOS26 runner also exposes an Apple Paravirtual device without
SIMD reduction. A newer OS does not provide the missing execution capability;
no GPU graph was attempted. [Original compiler diagnostic](../results/verification/metal-ir-20261116/initial-diagnostic.json)
and [device probe](../results/verification/metal-ir-20261116/macos26-device.json).

The corrected within-module IR job matches all four forwarding wrappers after
cancelling the exact coordinate shuffle. Each wrapper calls the same named
arithmetic helper as its original kernel.
[Within-module result](../results/verification/metal-ir-20261116/forwarding-diagnostic.json).
This comparison alone cannot establish agreement with the separately compiled
original library: helper bodies could be optimized differently across modules.

Extended the job to compile the unchanged original shader prefix separately.
The checker now compares reachable helper definitions across modules and all
original function definitions, in addition to wrapper forwarding. A missing or
changed helper cannot pass merely because its call site looks identical. Five
unit tests pass, including identical call sites whose helper body changes.
The original source prefix is checked locally to contain the complete pinned
kernel body and no added wrapper. This remains a textual compiler diagnostic;
metadata, later specialization and actual GPU behavior remain outside its proof.

The first separate-module run matches the four target wrappers and all reachable
arithmetic helper bodies. Of262 original functions,261 match; the remaining
compiler initializer is named from its source filename. Retained that incomplete
report. Added a unique-initializer name mapping that still compares its entire
body, rather than ignoring it. Six tests pass, including changed initializer
math and ambiguous extra initializer rejection.
[First separate-module report](../results/verification/metal-ir-20261116/separate-module-diagnostic.json).

### Alternative Q8 query grouping

Added a second locality candidate for the145MB Q8 output head. Each threadgroup
calls the unchanged pinned Q8 routine for two consecutive query rows, with a
uniform barrier before shared scratch reuse and an odd-tail guard. The original
arithmetic function, SIMD group count and scratch allocation remain unchanged.
The hypothesis is better nearby weight reuse with fewer threadgroups; repeated
dequantization, extra barriers and serialized work can erase any benefit.

It is default-OFF and exclusive with interleaving. The local runner's
`--group-q8` builds/gates a distinct candidate with ordinary/ngram controls and
optional suffix caps1/4/8. Marker checks reject mixed profiles. Host generation
and C++ compilation pass,448 dispatch cases and589896 split cases pass, and
logical row coverage checks2–9 rows including odd tails. The initial new fixture
test omitted its new analysis argument; corrected, that fixture passes in1.93
seconds. Default Release passes all7 suites in43.35 seconds, including the
expanded10-test gate/analysis set. Native shader validation remains pending.

A shared embedding helper avoids duplicating source expansion. The existing
interleaved shader remains byte-identical (SHA256bfde18aa…a17f3de), so this
refactor does not alter its compiled input.
[Host evidence](../results/verification/metal-q8-group-host-20261116/summary.json)
and [local grouped-Q8 command](metal-q8-group.md).

### Separate-module IR closure and Apple ARM CPU check

[The final separate-module run](https://github.com/VictorWong123/tokenomics/actions/runs/43822261855)
matches all262 original function definitions, including the uniquely renamed
filename-derived initializer, and all four interleaved wrapper/helper pairs.
[Retained diagnostic](../results/verification/metal-ir-20261116/separate-module-matched.json).
This remains a normalized textual compiler check; metadata, later function
specialization and actual M3 arithmetic/token gates are outside its scope.

Audited native pipeline routing: the pinned backend indexes actual compiled
`MTLLibrary.functionNames`, then resolves the requested base name from that
library. Added kernels do not require an enumerated name registry. Actual
pipeline creation and execution still require a supported Metal device.

Extended the existing bounded trained CPU workflow to select a standard public
Apple ARM runner, with the existing Ubuntu default preserved. It also retains
48 alternating sampler component timing pairs. The intended evidence is native
ARM sampler equivalence and learned CPU output/work agreement, not M3 speed or
GPU correctness. No optimization defaults changed.

### Full-head matrix component diagnostic

The [grouped-Q8 native run](https://github.com/VictorWong123/tokenomics/actions/runs/43825433600)
passes all10 stages: Apple ARM C++/Metal shader compilation, CPU fixture suites,
direct CPU row checks and Metal library initialization. GPU graphs remain
unavailable because the virtual device lacks SIMDgroup reduction.
[Retained native summary](../results/verification/hosted-metal-20261116/q8-group-native.json).

Added an optional, model-free M3 matrix study for the full896×151936 Q8 head
and three smaller quantized layer shapes. It compares plain short-MV,
interleaving and grouped-Q8 across1/2/3/5/9 rows, after exact one-row arithmetic
gates. Three rotated profile blocks retain2880 timing pairs and180 gates;
serial controls contain one-row matrix operations in one graph, rather than
full model decode. Reported component ratios cannot establish generation gains.
[Protocol and command](metal-matvec-component.md).

The small CPU harness passes20 gates/10240 bitwise comparisons in Release and
ASAN/UBSAN. The sanitizer build retains Q4-pair and greedy flags; this raw
harness invokes no sampler/model runtime. Missing Metal is rejected rather than
falling back to CPU. Three analyzer tests reject missing/duplicate records,
changed cross-profile data, failed gates, CPU results and invalid durations.
All7 default suites pass in43.55 seconds; Python/C++ formatting passes.
[Host evidence](../results/verification/matvec-component-host-20261116/summary.json).

### Native Apple ARM learned CPU result

[Run43826134066](https://github.com/VictorWong123/tokenomics/actions/runs/43826134066)
passes the native sampler equivalence executable, fixture orchestration, pinned
full-model SHA256 download, and all160 learned-model CPU exact-ID/stop/work
comparisons. The aggregate covers348 canonical output tokens per configuration:
ngram8 uses261 cycles/432 rows; suffix4 uses283 cycles/364 rows. Counts differ
from the x86 smoke; agreement is scoped to the same-machine ordinary reference,
and cross-ISA token agreement was not tested.

Retained the printed learned aggregate separately from the earlier synthetic
fixture summary in the same log. Full raw learned payloads remain in the
seven-day Actions artifact. Retained all48 sampler component timings: at151936
vocabulary entries, reference median146.6µs versus89.9µs, median paired ratio1.69.
These cache-warm component timings are not model throughput or an M3 result.
[ARM evidence](../results/verification/trained-arm-cpu-20261116/trained-summary.json).

The matrix component executable also compiles natively on the standard Apple
runner and passes its small CPU fixture under the grouped-Q8 build. All11
preflight stages pass in [run43827839125](https://github.com/VictorWong123/tokenomics/actions/runs/43827839125).
Large GPU matrices remain unexecuted; the device capability limitation remains.

### Incremental generated-source dependency fix

Found that plain `file(READ)` inputs were absent from CMake's regeneration
dependencies. Editing an owned shader include followed by a normal build could
therefore leave the embedded shader stale. Added a shared tracked-read helper
to CPU/Metal source generators. Reads of already-generated intermediate files
remain ordinary reads, avoiding treating output as an independent source input.

A reversible probe appends a comment to the owned grouped-Q8 shader, invokes
only `cmake --build`, and verifies automatic reconfiguration plus a changed
embedded shader hash. Restoring the source and building again restores the
generated shader byte-for-byte. The upstream submodule remains untouched.
All7 default suites pass in43.56 seconds and both Metal host suites pass.
[Probe evidence](../results/verification/source-regeneration-20261116/summary.json).

### Shared-load Q8 arithmetic prototype

Added an unselected shader prototype that shares integer weight loads and
conversions across two query accumulators, preserving the original Q8
eight-value accumulation and subsequent block-scale placement. It retains the
pinned reduction helper and a uniform barrier before scratch reuse. No main
runtime flag or dispatch selects this prototype yet.

The host test compiles the actual candidate source and the original function
extracted from the clean pinned shader. A capture stub replaces only the final
Metal reduction, exposing per-thread partial sums. Each of strict and native
FMA-contracted Release/ASAN variants passes2016 cases/3311616 exact partial sums,
covering row counts2–9, widths32–4864, three SIMD-group counts, padded strides,
zero/extreme half scales, cancellation and tiny activations. This does not
emulate Metal reduction or synchronization. A one-unit weight-conversion mutant
is rejected; the source and correct executable are restored byte-for-byte.
[Evidence](../results/verification/q8-shared-partials-20261116/summary.json).

The bounded compiler diagnostic now appends the unselected prototype, tests the
source-derived partials on native Apple ARM CPU, and retains its compiler IR.
The original/interleaved IR comparison explicitly lists additional functions
outside its semantic comparison. This avoids presenting successful compilation
of a new kernel as proof of its GPU arithmetic. [Prototype notes](metal-q8-shared-prototype.md).

### Shared-Q8 opt-in integration

[Native prototype run43830852115](https://github.com/VictorWong123/tokenomics/actions/runs/43830852115)
passes Apple ARM strict/contracted partial checks and Metal AIR compilation.
All262 original functions and four interleaved wrappers still match their
separate original module. The two shared helper/kernel additions are explicitly
outside that comparison. Their AIR shows one integer conversion feeding both
query recurrences; `fast` flags permit further GPU optimization, so this is not
a final GPU equivalence proof. Selected exact AIR bodies and native host counts
are retained with the prototype evidence.

Integrated `TOKENOMICS_METAL_Q8_SHARED=ON` as a default-OFF variant requiring
Q8 grouping, short-MV and embedded Apple shaders. The same row selector/grid
uses `_shared2` instead of `_grouped2`. The trained local runner's `--share-q8`
selects it and preserves the original ordinary/ngram controls. Profile gates
exclude other kernel variants and accidental greedy-sampler mixing. This also
closes a profile-labeling gap in the older short-MV validators.

The model-free component study now rotates four profiles through four blocks,
retaining5120 timing pairs/320 gates. It compares serial grouping against actual
shared weight work. Host C++ generation/compilation passes,448 dispatch and589896
split checks pass, all12 profile/orchestration tests pass, and all7 default suites
pass in45.79 seconds. Strict/contracted Release and ASAN partial checks also pass
with an added finite-value/poison-read rejection. The full native runtime profile
is pending; actual M3 graphs/tokens/timing remain unmeasured.
[Integration host evidence](../results/verification/q8-shared-integration-20261116/host-summary.json).

The first full native shared-Q8 run compiles the runtime, both direct harnesses
and both Metal shader sources. Eight native suites pass, including dispatch,
split geometry and runtime fixtures. The ninth, Python tooling, reaches the
240-second per-suite timeout without a reported assertion failure. Retained the
failed run; later backend/source-partial/library stages were not reached.
Raised that bounded suite allowance to600 seconds for the hosted ARM runner;
the workflow remains capped at45 minutes. This is a timeout correction, not a
numerical failure being waived.

The corrected [full native shared-Q8 run](https://github.com/VictorWong123/tokenomics/actions/runs/43835253840)
passes all14 stages, including nine native suites, direct CPU rows, component
fixtures, source-derived strict/contracted partials and Metal library compilation.
Its executable is byte-identical to the earlier timeout run. GPU graph execution
remains unavailable. [Native receipt](../results/verification/q8-shared-integration-20261116/native-compilation.json).

### Frequency-filtered copy fit investigation

Predeclared a24-setting causal frequency-copy grid with five caps in commit2cf2be3.
It uses only observed request-local suffix continuations, joint prefix-support
ratios, deterministic recent-occurrence ties and shorter-context backoff. Five
unit tests include18360 binary sequences checked against independent prefix
enumeration. The replay re-audits15 measured upstream aggregate counts per
operating point (30 total); no detailed measured cycles exist in the minimal
archives. Added matched raw-copy controls to separate filtering from lowering
the minimum match. No validation or reserved held-out trajectories are read.

Across192 M3 fit tokens, minimum-three frequency filtering at cap8 gives133
cycles/23 wasted rows, versus raw minimum-three133/27 and upstream ngram133/25.
The existing minimum-five cap4 point remains142/1. On Linux3B, minimum-two
frequency/cap2 gives141/13, versus raw141/17 and upstream ngram cap4 at142/34.
No new grid cell improves the zero-waste minimum-five cap1 point on either
operating point. Larger cycle savings require substantial rejected work. These
fit-only counts do not establish throughput or generalization, so no additional
runtime drafter is selected. Retained all120 grid cells/operating point and46905
computed proposal/cycle records, with query-shape histograms and Pareto aliases.
[Protocol/results](frequency-copy-fit-protocol.md).

### Broader per-row arithmetic audit

Audited normalization and elementwise dispatch beyond matrix/softmax. Pinned
RMS-normalization reduction geometry depends on width/pipeline capacity, not
query-row count; no normalization patch is needed. Unary contiguous dispatch
changes at32768 elements, crossed by4864-wide FFN batches between six and seven
rows. Expanded the raw backend gate with96 normalization/activation and160
absolute-position RoPE cases, including both head counts2/14, normal/NeoX layouts
and position prefixes0/248/255/511. These nodes stay unmarked, matching the actual
owned graph scope. Full-model exactness remains a separate gate.

The expanded804-case gate passes4181696 bitwise value comparisons in Linux
Release and ASAN/UBSAN. The sanitizer raw-backend profile retains Q4-pair and
greedy flags; no sampler/model runtime is invoked. All115 Python tooling tests
report success with8 skips in38.783 seconds. Experiment source fingerprints now
include verification files and nested CMakeLists; changing a gate during a run
is detected, while older receipts retain their original scope.
[Coverage and local command](backend-row-coverage.md).

### Backend-sampling substitution check

Audited the pinned experimental backend sampler as a possible way to reduce
host logit transfer/scanning. Its argmax contract differs from the original
strict-first host sampler. A direct actual CPU-backend probe reports24/27
mismatches in both Release and ASAN/UBSAN; the384-case host sampler equivalence
suite still passes. A finite151936-entry example with equal maxima at IDs0/1
selects0 in the reference and1 in GGML argmax. The Metal source also reduces
equal partial winners using maximum IDs, but no GPU result is claimed.

The backend temp-zero step compresses to one candidate/logit through argmax;
removing it on that path would retain full sampled-logit copying. Thus the host
no-temp optimization cannot simply be combined with backend sampling. Left
backend sampling disabled and added an optional model-free semantic probe,
avoiding any gather from an invalid argmax index. Native preflight now records
this diagnostic's mismatches explicitly rather than treating completion as
proof of substitution. [Contract and reproducer](backend-sampling-contract.md).

Extended bounded model-header inspection to select the already-pinned1.5B
public model. It retains metadata only and uses no Codespaces or model-weight
artifact upload. This will identify which existing kernel paths matter at that
larger operating point before extending the local study.

### 10:20 UTC — strict raw-logit prototype and second-model metadata

The direct host greedy prototype avoids materializing the 151,936-element
`llama_token_data` array while preserving strict-first comparisons. The pinned
reference agrees on 384 existing and 349,524 exhaustive tie/nonfinite rows in
Release and ASAN/UBSAN. Runtime integration remains disabled pending native
Apple ARM component checks. Local x86 component observations are retained;
no model-throughput benefit is inferred from them.

Public hosted metadata run 43845958766 inspected only the pinned 1.5B GGUF
header: 28 layers, hidden width 1536, FFN 8960, 12 query/2 KV heads and Q6_K
output projection. The header confirms that the Q8-specific prototype does not
optimize this model's output projection. Full weights were not downloaded or
SHA-verified by that metadata job. The receipt explicitly records this limit.
