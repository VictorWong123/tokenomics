# Dedicated Codespace fixture controller

Private trusted-writer infrastructure: pushes to `work` and `codex/**` run fixed
controller code from the triggering commit. There are no PR or manual triggers.
This replaces—not supplements—the self-hosted listener. Retire that listener
before enabling the controller; the dedicated machine must initially be Shutdown
and must not be used interactively during an owned run. Workflow concurrency
serializes this repository's controllers; admission rejects an already active
machine and the remote helper rejects concurrent operations.

Configure a fine-grained `CODESPACES_TOKEN` restricted **only** to
`VictorWong123/tokenomics`, with Codespaces read and Codespaces lifecycle admin
write. Store it as the GitHub Actions repository secret, never in Cloud or the
Codespace. Cloud never receives the secret. Set repository variable
`TOKENOMICS_CODESPACE` to the exact named two-CPU private Tokenomics machine.

The hosted checkout uses the read-only GitHub token and complete history to
create a credential-free exact-commit bundle. Authenticated `gh codespace cp`
transfers it; no PAT or credential is sent remotely and no private-origin fetch
is attempted. The original `/workspaces/tokenomics` checkout is only inspected
for its origin. The helper creates its own bare repository and detached worktree
under `/workspaces/.tokenomics-operations/<run>-<attempt>`. Only the pinned public
llama.cpp submodule initializes normally. The machine must provide `/usr/bin/python3` version 3.12 with venv/ensurepip support,
`git`, `timeout`, and the existing fixture build prerequisites. The helper checks this before building and uses PATH
`/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin`, keeping system Python first, allowing `/usr/local/bin/git`, and excluding runner toolcaches. No container
rebuild, idle-timeout change, model download, or core pilot is performed.

The fixture command has a 20-minute deadline; the hosted job has a 45-minute
limit reserving time for transfer, cleanup and stop polling. Catchable failures
retain a nonsecret controller report/log and always attempt owned-path cleanup
and actual Shutdown confirmation. Failed cleanup or unconfirmed Shutdown fails
the job. Forced runner termination or service outage can defeat in-process
cleanup; inspect the dedicated machine before retrying. No unrelated active
machine is adopted or stopped.

The bounded archive contains only root `status.json`, `environment.json`, five
fixture-pipeline files (`complete.json`, `correctness.jsonl`, `cycles.jsonl`,
`requests.jsonl`, `run_manifest.json`), and `execution-manifest.json`. Digest,
member names/types/sizes, JSON syntax and manifest identity are checked before
acceptance. The manifest records exact source SHA, host, run/attempt and bundle
digest; the bundle itself is never uploaded. Artifacts expire after three days.
No credential, runner log, model, build tree, venv or broad results directory is
collected. The controller log contains only fixed phase summaries, not remote
command output. Run unit tests with `python3 -m unittest discover -s
scripts/codespaces -p 'test_*.py'`.
