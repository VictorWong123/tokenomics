# Trained-model divergence investigation

The original trained-model correctness gate failed. The consistent-CPU-kernel
fix and test deadline correction are now verified by a complete Linux run:
all 272 exact-token comparisons and all 14 diagnostics passed in
[42420516268](https://github.com/VictorWong123/tokenomics/actions/runs/42420516268).
No feasibility or speedup claim is supported. The authorized scope ends after
Linux testing and shutdown; do not continue to performance or controller work.

Codespace run [42403427793](https://github.com/VictorWong123/tokenomics/actions/runs/42403427793)
used source `54b4252cb4f2642156eaf7d4be71a1f945cc77c4`, two CPU threads,
Qwen2.5 0.5B draft and 1.5B target Q4_K_M. After seven comparisons it stopped
on `retain-fixed4`, `development-qa-00`, repetition 0, zero-based token 29:
ordinary target token 16417 versus candidate token 2415. Model SHA256 values:

- Target: `6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e`
- Draft: `74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db`

The evidence archive SHA256 is
`c79cee3a603201f71e11f5301c879d47b24a94ce2dc1d4502c3e74574714ebd5`.
The finalizer confirmed actual Codespace Shutdown. This establishes model
download access and a reproducible correctness failure, not project success.

The parent coordinator independently inspected earlier artifact 13586471641
from run 42399562084: the candidate batch timed out at 1800 seconds, but retained
243 of 272 rows, including 78 exact-token mismatches. The zero comparison count
in that older summary reflected delayed batch comparison, not absence of results.
Its first fixed4 mismatch agrees with the incremental rerun; fixed6 first differed
at token 8. Keep that original artifact; do not replace it with the retry.

## Targeted diagnostic contract

`scripts/diagnose_models.py` runs the same development prompt for 32 output tokens
with two resets per configuration: ordinary target-only, resident target-only,
and retain/upstream cache modes with caps 1, 4, 6. It checks identity, repetition,
output cap and execution settings against each submitted job. Comparisons match
the reference repetition. Cycle/cache invariants remain checked.

The opt-in `diagnostic_logits` job field records raw target top-two IDs and logits
before calling the unchanged upstream verifier. Only committed rows correspond
to canonical output prefixes. Probe times include diagnostic overhead and must
never be used as performance measurements. Normal jobs leave diagnostics off.

Fresh checkpoint replays at tokens 8, 29 and observed divergent cycle beginnings
prefill both resident contexts from the same canonical token IDs for every action
0, 4, 6. They use one cycle and the same output budget. Differences between these
actions would support a batch-shape effect; agreement would not establish that
the original incremental cache was correct. Reconstructed prefixes do not retain
the original computation's floating-point history. Top-logit margins and agreement
between raw top1 and emitted tokens are evidence, not a tolerance that excuses
an exact-token failure. Repeated references check reset stability.

The fixed Codespace workload runs this bounded diagnostic before
the full gate. `diagnostic_complete` only means evidence collection completed;
it explicitly carries `correctness_supported: false`. The artifact includes all
submitted jobs, token records, cycle traces, manifests and stderr. Lifecycle
ownership, existing resource selection and shutdown logic are unchanged.

Local synthetic validation exercises this diagnostic path, including incorrect
record matching and uncommitted-row exclusion. Synthetic agreement cannot resolve
the trained-model failure. This was the evidence available before the completed
native diagnostic and correction recorded below.

## Cancelled diagnostic attempt

Run [42404730880](https://github.com/VictorWong123/tokenomics/actions/runs/42404730880)
at `fd37f5b3fed054010252f91dae122d56e5adbf57` was cancelled at
2026-11-07 19:17:54 UTC during model verification, before the diagnostic stage.
It did not reach the job deadline. No evidence archive was uploaded (artifact
count zero). The finalizer explicitly confirmed Shutdown at 19:19:01 UTC.
The cancellation's initiator/reason is not established by the inspected metadata;
no replacement run was launched automatically.

Follow-up preparation on `review/divergence-followup` records selected CMake
switches and checks committed logit mapping on random fixtures. Native CPU,
`GGML_CPU_REPACK=OFF`, and additionally `GGML_LLAMAFILE=OFF` builds each passed
four local CTests. Native and no-repack synthetic diagnostics each completed
14 comparisons and six fresh-prefix probes. These checks establish usable
build options, not a trained-model fix. The pinned CPU repack implementation
switches from GEMV to GEMM at four rows; that is a testable source-based hypothesis,
not an established explanation. Obtain the cancelled diagnostic's replacement
results before selecting a backend ablation or making a root-cause claim.

## Completed native diagnostic and candidate correction

Run 42405912017 at `9a660860cdbca5cde5e74036e35ceb48ee75c669` completed the
diagnostic and confirmed Shutdown. Both reference repetitions were identical.
Retain and upstream cache modes reproduced the same fixed4/fixed6 mismatches
in both repetitions; the raw top-one logit IDs matched emitted tokens.
At fixed4 token 29, ordinary decoding favored 16417 by 0.6522 logits while
verification favored 2415 by 0.1184. Fresh identical-prefix replays also
changed logits with batch width, showing numerical batch dependence without
prior incremental history. This supports a CPU arithmetic explanation; it
does not independently prove every cache-content invariant.

The candidate default `TOKENOMICS_CPU_BATCH_INVARIANT` configuration disables
weight repacking, llamafile SGEMM and BLAS. The pinned tiled K-quant path is
excluded from the at-most-nine verification rows, while larger common prefills
retain tiling. Verification uses consistent dot kernels across widths. The full 272-case
trained-model gate, unchanged exact-token comparisons, and explicit shutdown
must pass before this correction is considered verified. The generated unit
fixtures now also cover quantized weights; these are not trained-model proof.

The first correction run, 42407413173 at `8d796fd7625f7ddc676ef4106b99dcbeec345251`,
completed all 14 diagnostics without a mismatch and 249 gate comparisons without
a mismatch. It then hit the 3,600-second candidate timeout during the repeated
2,048-token boundary cases. Evidence artifact 13589584594 was retained; actual
Shutdown was confirmed and the workflow automatically created issue #7.
The follow-up restores tiled larger prefills while protecting all supported
generation widths. Exact-token comparisons, all 272 cases, context size and
deadlines were unchanged in that follow-up. Q4_K fixtures now exercise large-prefill/small-decode
transitions and exact first-row top-two logit equality across proposal caps.


Run 42414433936 at `d11a1fd79f316ebd619eead6a924e40fbdad6679` again passed all
14 diagnostics and reached 258 passing comparisons, with no mismatch, before
the same 3,600-second candidate deadline. The fixed Linux integration-test
budget is now 7,200 seconds per subprocess, inside the existing 9,000-second
child and 210-minute job limits. This changes no cases, contexts, models,
comparison rules or CPU profile. All 272 cases still have to finish.

## Verified resolution

The current diagnostic schema is 2. It preserves the original remaining token
budget at each fresh-prefix checkpoint, verifies exact prefix IDs and summarizes
every committed row, rather than row 0 alone. Each row records whether all
preceding emitted tokens still match the canonical continuation; a matching
later token after divergence is not evidence of recovery at the same history.
The earlier schema-1 fresh-prefix summaries only covered their first row.
Their limitation does not invalidate the separate full 272-token-comparison
gate, and fresh-prefill probes still do not reproduce the original KV history.
Cloud regressions include first-row agreement followed by a row-1 mismatch.
A controlled execution checks 14 full comparisons and 20 committed replay rows
across six jobs; checkpoint 29 retains three output tokens of the original 32.
See `results/verification/cloud-20261108/all-row-diagnostic.json`.

Run 42420516268, attempt 1, on `main` source
`d1bb6fb778695515614d21cf78959f63f544db7b` completed on dedicated Codespace
runner 22. The 14 diagnostic comparisons had no mismatch, reference resets
were equal, and the six fresh-prefix probes matched emitted tokens. The full
trained-Qwen gate then passed 272/272 comparisons with the target/draft hashes
listed above. Zero/partial/full acceptance, catch-up, switching, budget stops
and context stops were observed. The artifact uploaded and the finalizer
confirmed actual Shutdown. Issues #6 and #7 are closed with this evidence.

The native mismatch depended on batch width; selecting consistent CPU kernels
for the supported generation widths removed it in the complete unchanged
matrix. The later failures were test deadlines, not observed token mismatches;
the 7,200-second subprocess allowance let the two-core integration test finish.
See [handoff](handoff.md) for artifact identity and verification limits. No
performance experiment or project continuation followed the successful tests.
