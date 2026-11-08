# Tokenomics continuation handoff

## Current continuation (8 November 2026)

The user requested completion of the broader project and clarified: develop
and check in Codex Cloud, push checked stages to `main`, then execute trained
tests and measurements in Codespaces. This supersedes the earlier test-only
scope recorded below.

`87c7811` reuses C++ proposal/batch storage, reserves bounded outputs and avoids
retaining discarded minimal logs. It also prevents held-out baseline retuning,
rejects incomplete/duplicate measurement grids, and adds hash-checked resumable
pilot groups plus the fixed experiment lifecycle. Cloud Release/sanitizer tests
pass, including leak detection; the fixture suite now has 330 cases plus 25
minimal/checkpoint comparisons. All 168 fixture requests pass exact IDs.

Trained timing run 42452778183 passed nine comparisons and confirmed Shutdown.
Its three-prompt, one-repeat screen favored target-only: fixed4/target throughput
ratio 0.501; confidence/target 0.533. These are preliminary observations, not
Q1/Q2 decisions. The bounded repeated pilot is run 42454095789 at exact source
`87c7811ed5fe24024f50782a1307ecbcef52bdee`. It paused successfully after 11/108
blocks, preserving 110 passing exact-ID checks. Artifact ZIP, inner archive and
all 636 members were hash-verified; hosted finalize confirmed actual Shutdown.
Every measured drafting-policy cell was slower than its ordinary reference.
This incomplete, unbalanced prefix has no full-suite estimate or gate decision.
The raw aggregates and provenance are in
`results/verification/codespace-partial-20261108`. Preserve the source on resume;
do not combine different binaries/protocols.

That pilot is preserved as `experiments/pilot-87c7811` for reproducibility.
Current Pilot B schema 3 restores original target KV and identical prepared
draft KV from memory-bounded, within-process checkpoints. It includes the
dynamic baseline and rejects a different binary/model/execution setting.
Older points remain exploratory; freeze a new checkpoint-capable driver for
confirmation. The optional stock diagnostic and clean-submodule build check
also pass in Cloud.

The latest CLI also repairs incomplete UTF-8 display text instead of crashing
JSON serialization. Eighteen byte-boundary cases preserve exact IDs across
target/fixed/script modes; Release and targeted sanitizer checks pass. This
changes the driver binary, so do not mix its measurements with the frozen
`87c7811` pilot or bypass replay's binary check. Retain the older run as
exploratory evidence; freeze the corrected driver for future confirmation.

Diagnostic schema 2 also covers every committed replay row and the original
remaining budget. A row-0 match no longer hides a row-1 divergence, and rows
after divergence are marked as having a different history. The Cloud fixture
diagnostic passes 14 full comparisons and checks 20 replay rows; its
`correctness_supported` remains false, as diagnosis is not the trained gate.

The saved-state implementation passes 378 C++ differential cases, plus the
25 minimal/checkpoint and 18 Unicode cases. First-row logits match the original
trace exactly across all fixed caps after long-prefix state restoration.
All four sanitizer suites passed with leak detection, and the extended-prefix
runtime check passed separately. The fixture Pilot B checks 264 canonical
action outputs across 12 checkpoint groups. Normal performance conditions
cannot request snapshots or diagnostic logits. This is fixture evidence;
trained-state validation is part of the next Codespace timing screen.

Current warm-up uses a frozen short prompt, and budget estimation separates
fixed request costs from actual-token decode cost. The larger candidate can
now be screened with 5 GiB available and must still pass the measured 20%
headroom requirement. Select it with `candidate=qwen25-05-3-q4`; no new machine
or lifecycle credential route is introduced.

The experiment workflow now supports fixed `drafter=ngram` timing/pilot work
without downloading or loading a draft model. Cloud integration checks execute
its 18-request cap sweep and original-state replay. A period-nine transformer
fixture forces real lookup proposals at every cap, with rejected proposals,
fallback/re-entry and resets; the suite now reports 390 differential cases.
Release and extended sanitizer runtime checks pass with leak detection. These
new 12 comparisons are fixture evidence, not a trained-model result. Relevant
paper/source comparisons are recorded in `docs/related-work.md`.

Run 42461309986 at `1f945147` completes the 0.5B→3B model screen: all 21
comparisons and 66 original-state checkpoint actions pass. All 173 artifact
members are hash-verified; the rechecked raw data and analysis are retained in
`results/verification/codespace-3b-20261108`. Every fixed cap loses; the best
screen ratio is fixed1 at 0.768 (confidence 0.682). Peak screen RSS is 2.75 GiB.
Hosted finalization confirms actual Shutdown. Three fit-only prompts cannot
clear Q1/Q2; the full-pilot duration extrapolation is 29.67 hours.

Run 42461982414 attempt 2 completes the cheaper ngram screen at `6a4eea1`:
18 exact comparisons plus 66 original-state action checks pass; all 134 archive
members are verified and actual Shutdown confirmed. Fixed1 is near-neutral at
1.0113; every larger cap loses. Attempt 1 failed startup before inference;
the unchanged owned lifecycle succeeded on retry. Raw data are in
`results/verification/codespace-ngram-20261108`. The fit evidence freezes fixed1
for a targeted nine-prompt development confirmation (128/512 caps, three
repetitions, target/fixed1/fixed8). This does not clear full Q1/Q2 or use held-out
prompts. `confirmation` is a fixed workflow phase; preserve its exact source
on resume. Its driver/model/settings must match the fit point. Tests cover a
complete 108-request fixture grid and reject partial grids or validation retuning.

Still required: complete Q1/Q2 on resource-safe candidates; implement a justified
cost-aware or narrower controller; quantify instrumentation/probe overhead;
freeze development priors and baselines; run held-out evaluation/ablations;
finish the measured report/demo. Stock-executable stopping differs from the
driver; an executed synthetic diagnostic confirms this difference. Mac/Metal is supported by
configuration but unexecuted. The full runtime project remains incomplete.

Evidence-based wording now: “Building a portable C++ speculative-decoding
runtime with checked KV rollback and lazy synchronization; validated with
390 transformer fixture cases and a 272-comparison trained-Qwen Linux gate.”
Do not claim acceleration from the current unfavorable timing screen.

## Earlier completed Codespace setup

The requested Codespace setup and Linux test repair are verified. Canonical
source is `main`; PRs [#4](https://github.com/VictorWong123/tokenomics/pull/4) and
[#8](https://github.com/VictorWong123/tokenomics/pull/8) are merged. Do not resume
the earlier project continuation, timing, pilots, controller work or training
without a new instruction. The continuation above supplies that new instruction.

## Completed verification

[Run 42420516268](https://github.com/VictorWong123/tokenomics/actions/runs/42420516268),
attempt 1, completed successfully at test source
`d1bb6fb778695515614d21cf78959f63f544db7b` (source tree
`bf3bf3d7189ff8ac729b38c99f8b2139e847c754`). Later documentation-only commits do
not change that tested implementation.

- Hosted initialize succeeded and claimed the intended Codespace.
- The workload ran on self-hosted runner 22,
  `tokenomics-codespace-long-tasks`, in Codespaces.
- Both trained Qwen2.5 Q4_K_M model hashes matched; all 14 diagnostics and all
  272 exact-token comparisons passed. Reset references matched. Observed coverage
  included zero/partial/full acceptance, catch-up, switching, budget and context
  stops.
- Artifact `tokenomics-model-check-42420516268-1` (ID 13596331570) uploaded.
  Its GitHub ZIP digest is
  `7a91126e3627b34d1bf4a4e08e88e5abb568462f474ed94442e2d4c73f9564d5`;
  the workload's inner archive SHA256 is
  `d20a9dff6ba877b1f3d540279a947eabee55d55364dc4dcf53ea21ae54e16235`.
  Artifact retention ends 2026-11-11; the run logs retain the compact reports.
- Hosted finalize succeeded and explicitly confirmed actual Codespace Shutdown.
  Its `fixture_evidence_valid: false` is expected: it performs lifecycle cleanup,
  while the separate model summary proves the tests passed.

The model summary, initialization/finalization reports, run/job metadata and
artifact metadata were checked from Codex Cloud. The artifact has now also been
downloaded: ZIP, inner archive and all 32 manifest members match their recorded
hashes. All 272 candidate identities/settings, exact IDs/stops and cycle/cache
accounting were independently rechecked. The full small raw archive is retained
in `results/verification/codespace-gate-20261108`. Its old schema-1 diagnosis
does not validate current original-KV replay. Reuse the completed evidence;
no setup rerun is needed.

## Future Linux tests

Use a trusted `main` push or the fixed **Run workflow** button on
[Codespace Linux verification](https://github.com/VictorWong123/tokenomics/actions/workflows/codespace-fixtures.yml).
Markdown-only pushes are ignored. The dedicated existing resource is
`tokenomics-long-tasks-jwgxp7j7xvqfq76w`; the old Codespace is not used.
Cloud access is through Actions initialize/workload/finalize.
See [Cloud route](../.codex/cloud.md) for monitoring and credential boundaries.

Qwen2.5 0.5B draft and 1.5B target weights persist at
`/workspaces/.tokenomics-model-cache`, outside checkout cleanup, and are
SHA-verified before tests. Missing weights are downloaded automatically. Neither
the user's computer nor a local model server is needed.

Failed tests or lifecycle stages create/update a deduplicated GitHub issue
containing the run, source, available failing case and shutdown result. Only a
short hosted reporting job has issue-write permission. Earlier timeout runs
proved this reporting path. Resolved issues [#6](https://github.com/VictorWong123/tokenomics/issues/6)
and [#7](https://github.com/VictorWong123/tokenomics/issues/7) retain the failures
and the successful resolution. Issue #3 concerns later work and remains open.

The repaired CPU configuration uses consistent kernels for supported generation
widths and tiled larger prefills. The fixed test permits 7,200 seconds per
candidate subprocess within the unchanged 9,000-second child and 210-minute job
limits. All original 272 comparisons, model hashes, 2,048-token context and
exact-token rules remain. See [investigation](divergence-investigation.md) and
[test contract](model-correctness.md).

This establishes the bounded Linux test route and its trained-model checks.
Performance, Mac execution, stochastic sampling and completion of the broader
research project were not established by that verification.
