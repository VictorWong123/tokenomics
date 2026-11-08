# Cloud setup and test route

Canonical source is `main`. Fresh Cloud tasks should fetch and inspect
`origin/main`, preserving dirty or unpublished work. `work` retains the earlier
setup baseline.

## Project continuation (8 November 2026)

The user has now explicitly requested completion of the remaining project,
including optimization and Codespace execution. This supersedes the earlier
test-only stopping point below. Preserve the existing lifecycle and credential
boundaries. Develop and check changes in Codex Cloud, push checked stages to
`main`, then use Codespaces for trained-model tests and measurements.
`codespace-experiment.yml` adds manual fixed `timing` and resumable `pilot`
tasks on the same dedicated resource and global concurrency group. Experiments
do not repeat the completed 272-comparison gate. Partial pilots are retained by
source SHA outside checkout cleanup; dispatch that exact SHA again to resume.
The worker pauses between complete blocks/checkpoint groups before its budget,
uploads bounded evidence and always finalizes. Timing/pilots use development
prompts; held-out evaluation and controller implementation remain gated.

Pilot run 42454095789 is frozen at `87c7811` (also preserved as branch
`experiments/pilot-87c7811`). Resume that branch to retain its Pilot A blocks,
even after main advances. Its schema-1 Pilot B must not justify a controller:
the strengthened schema-2 replay also compares against the existing dynamic
baseline. Once Pilot A completes, the standalone `python -m bench.replay`
command can run schema 2 from those traces, requiring identical executable and
model hashes plus context/batch/thread/backend settings. Use a new replay
directory and retain both versions. New main pilot runs use schema 2 directly.

## Verified Linux test route

[Run 42420516268](https://github.com/VictorWong123/tokenomics/actions/runs/42420516268),
attempt 1, passed at source `d1bb6fb778695515614d21cf78959f63f544db7b`.
Hosted initialize, the dedicated Codespace workload (runner 22), artifact upload
and hosted finalize all succeeded. `TOKENOMICS_MODEL_SUMMARY` reports
`state: passed`, 272/272 exact-token comparisons, 14 diagnostics without
mismatch and the expected trained Qwen hashes. The finalizer separately reports
`shutdown_confirmed: true`. Issues #6 and #7 are resolved. Reuse this evidence
rather than rerunning setup. See [handoff](../docs/handoff.md) for artifact
identity and verification limits.

Qwen2.5 0.5B draft and 1.5B target Q4_K_M weights persist in
`/workspaces/.tokenomics-model-cache`, outside the clean Actions checkout.
Every test verifies their SHA256 hashes; missing files are downloaded there.
The authorized scope is Linux tests only. Stop after testing and shutdown;
do not proceed to timing, pilots, training or project development.

## Running and monitoring tests

The supported Cloud → Codespace route is a trusted `main` push or the fixed
**Run workflow** button on
[Codespace Linux verification](https://github.com/VictorWong123/tokenomics/actions/workflows/codespace-fixtures.yml).
Markdown-only pushes are ignored. Runs start billable Codespaces compute, so
obey any hold before triggering. There is no arbitrary command input.
The lifecycle secret stays in short hosted initialize/finalize jobs.
The long workload runs on the dedicated self-hosted Codespace runner.
Target/runner identities are repository variables.

Cloud access is through this Actions lifecycle, not an interactive Codespaces
connection. An empty `gh codespace list` does not test that route: the Cloud
repository credential differs from the hosted lifecycle secret. Do not retry
managed Cloud `/user/codespaces` calls, transfer PATs, direct-SSH, add Tailscale
or replace credentials to obtain direct access.

To inspect runs with a valid CLI credential:

```bash
gh run list --repo VictorWong123/tokenomics --workflow codespace-fixtures.yml
gh run view RUN_ID --repo VictorWong123/tokenomics --json headSha,status,conclusion,jobs
```

If the short-lived Cloud CLI credential expires (HTTP 401), use connected
GitHub tools for repository Actions run/job reads and decoded job logs.
A rerun appears as another attempt of the same run, not a new list entry.

Require successful initialization, workload, artifact upload and finalization.
Read `TOKENOMICS_MODEL_SUMMARY` in the workload log: correctness requires
`state: passed` and all 272 comparisons; `diagnostic_complete` alone is not a
correctness pass. Read `TOKENOMICS_CONTROLLER_REPORT` in the finalizer log and
require `shutdown_confirmed: true`. A workload summary alone does not prove
shutdown. The model finalizer intentionally performs cleanup without fixture
validation; its `fixture_evidence_valid: false` does not negate model evidence.

The historical fixture workflow additionally emitted
`TOKENOMICS_EVIDENCE_SUMMARY`, gated by 24 blocks / 168 synthetic records.
That fixture proof is separate from the current trained-model contract.
The workload job has a 210-minute budget, with bootstrap and child deadlines.
Preserve ownership checks, runner identity, secret isolation and always-cleanup.

Failed initialization, tests or cleanup create/update a deduplicated GitHub
issue with the source, run, available failing case and shutdown confirmation.
Only the short hosted reporting job has issue-write permission; the Codespace
has none. Automatic issue creation/update was exercised by the timeout runs;
reporting was skipped on the successful verification.

The bounded Actions artifact is detailed evidence. Cloud may read logs while
artifact ZIP storage is blocked; do not bypass the proxy. Use the summary/logs
and the GitHub UI for the archive when needed.
