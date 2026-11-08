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
`87c7811ed5fe24024f50782a1307ecbcef52bdee`. Its eventual artifacts and shutdown
must be checked. Preserve that source on resume; do not combine new-code rows
with its frozen run. Details and retained evidence are in `feasibility.md`.

That pilot is preserved as `experiments/pilot-87c7811`. Resume that exact branch
for Pilot A, then use schema-2 replay from main before a controller decision.
Schema 2 adds the existing dynamic baseline and refuses a different binary,
model or execution setting. See `experiment.md`; schema-1 Q2 cannot clear the
strengthened gate. The optional stock diagnostic and clean-submodule build
check also pass in Cloud.

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

Still required: complete Q1/Q2 on resource-safe candidates; implement a justified
cost-aware or narrower controller; quantify instrumentation/probe overhead;
freeze development priors and baselines; run held-out evaluation/ablations;
finish the measured report/demo. Stock-executable stopping differs from the
driver; an executed synthetic diagnostic confirms this difference. Mac/Metal is supported by
configuration but unexecuted. The full runtime project remains incomplete.

Evidence-based wording now: “Building a portable C++ speculative-decoding
runtime with checked KV rollback and lazy synchronization; validated with
330 transformer fixture cases and a 272-comparison trained-Qwen Linux gate.”
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
artifact metadata were checked from Codex Cloud. The artifact ZIP was not
independently downloaded in Cloud. Reuse the completed evidence; no setup rerun
is needed.

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
