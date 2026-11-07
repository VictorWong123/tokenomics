# Cloud setup and test route

Canonical tested source is `work`, not default `main` or the old `e089` seed.
Fresh Cloud tasks should fetch and inspect `origin/work`, then base `codex/**`
feature branches on the user-intended exact revision. Preserve dirty/unpublished
work; never reset it merely to follow the branch.

The supported Cloud → Codespace test route is a user-authorized trusted
`work`/`codex/**` push into `.github/workflows/codespace-fixtures.yml`. Pushes can
start billable Codespaces compute, so obey any hold before publishing. Do not
retry managed Cloud `/user/codespaces` calls, copy PATs, direct-SSH, or add
Tailscale. The lifecycle secret stays in short hosted Actions initialize/finalize
jobs. The long workload runs on the dedicated self-hosted Codespace runner.
Target/runner identities are repository variables, not names embedded here.

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
