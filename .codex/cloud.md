# Cloud setup and test route

Canonical source is now `main`. It includes the project and the Codespace test
workflow; `work` retains the earlier setup baseline. Fresh Cloud tasks should
fetch and inspect `origin/main`, then base feature branches on the intended
revision. Preserve dirty/unpublished work; never reset it to follow a branch.

The supported Cloud → Codespace test route is a trusted `main` push or an
authorized manual run of `.github/workflows/codespace-fixtures.yml`. Runs can
start billable Codespaces compute, so obey any hold before triggering. Do not
retry managed Cloud `/user/codespaces` calls, copy PATs, direct-SSH, or add
Tailscale. The lifecycle secret stays in short hosted Actions initialize/finalize
jobs. The long workload runs on the dedicated self-hosted Codespace runner.
Target/runner identities are repository variables, not names embedded here.

Cloud access is through this Actions lifecycle, not an interactive Codespaces
connection. An empty `gh codespace list` in Cloud does not test that route:
Cloud's repository credential is separate from the hosted lifecycle secret.
Do not replace credentials or transfer that secret to obtain direct access.
To find the latest verification across branches, use:

```bash
gh run list --repo VictorWong123/tokenomics --workflow codespace-fixtures.yml
gh run view RUN_ID --repo VictorWong123/tokenomics --json headSha,status,conclusion,jobs
```

Require successful initialization, workload, artifact upload and the finalizer's
`shutdown_confirmed: true`. Check the workload's own result contract: model
correctness requires `TOKENOMICS_MODEL_SUMMARY` with `state: passed` and 272
comparisons; `diagnostic_complete` alone is not a correctness pass. A rerun
appears as another attempt of the same run, not a new Actions list entry.
The Actions page exposes **Run workflow** for the fixed Linux verification.
Failed initialization, tests or cleanup file/update a deduplicated GitHub issue
with the source, run, failing case and shutdown confirmation. Only the short
failure-reporting job gets issue-write permission; the Codespace has none.

Read `TOKENOMICS_EVIDENCE_SUMMARY` in the final Actions log for validated compact
results and confirmed Shutdown. A workload summary alone does not prove shutdown.
The bounded Actions artifact is canonical evidence. Cloud may read logs while
artifact ZIP storage remains blocked: do not bypass the proxy. Use summary/logs,
and the user's Mac or GitHub UI for the canonical artifact when needed.

For an authorized future long task, edit only trusted checked-in self-hosted
workload code/step and its explicit expected result paths/allowlist; retain
initialize/finalize, runner identity, secret isolation and always-cleanup.
Review before triggering. There is no arbitrary command input. Fixture 24/168
proof gates only the fixture success line, never generic shutdown cleanup.
A custom task without that proof gets a cleanup-only lifecycle report; use its
own approved result contract and keep within the 210-minute workload budget.
