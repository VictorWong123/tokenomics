# Development workspace and test environment

## Names and responsibilities

| Name | Purpose | Current location / allocation |
|---|---|---|
| **Codex Cloud workspace** | Edit C++, run portable builds, fixtures, sanitizers and analysis; commit checked changes to `main` | `/workspace/tokenomics`; Linux, four-core quota, 16 GiB memory limit |
| **Codespaces test environment** | Run trained-model correctness tests and measurements after the source is pushed | Dedicated two-logical-CPU Linux Codespace; about 7.8 GiB RAM, no swap |
| GitHub-hosted CI | Short lifecycle jobs, portable source checks and verified-artifact retention | Ephemeral Ubuntu workers; separate from the measured Codespace |

Throughout this project, “the environment” means the Codespaces test environment.
Use “Codex Cloud workspace” when referring to the development shell. The chat
agent and that shell have separate connections: the chat can remain active when
its shell attachment fails.

Develop and check changes in Codex Cloud, push a checked stage to `main`, then
run trained tests in Codespaces. During a temporary Codex Cloud shell interruption,
connected GitHub tools and hosted CI preserved source, checked the prepared
comparison and verified/retained completed artifacts. Development resumed in
Codex Cloud when its shell attachment recovered. No Codespaces configuration
change was needed for that interruption.

Separate [Apple M3 results](../results/verification/mac-m3-metal-20261111/README.md)
cover a physical Mac with 8 GiB unified memory, macOS 15.6.1 and Metal offload.
The smaller Mac controller target and the Linux 3B target have separate calibrations.

## Supported Codespaces route

Use the repository's fixed GitHub Actions workflows. Their hosted initialize
job claims the existing dedicated runner with an exact run/attempt/source owner.
The self-hosted workload receives that checked-out source. Hosted finalize
verifies ownership, stops the Codespace and checks the actual `Shutdown` state.
All long workloads share `tokenomics-dedicated-codespace` concurrency.

The lifecycle credential stays in the short hosted jobs. Model tests do not
receive it. Runner ID/name checks, tmpfs leases, restricted file modes and the
disjoint checkout check protect the user's checkout. See
[the lifecycle documentation](../scripts/codespaces/README.md) and
[Cloud route](../.codex/cloud.md) for the complete boundary.

Models persist in `/workspaces/.tokenomics-model-cache`, outside Actions checkout
cleanup. Resumable pilot/confirmation measurements persist separately under
`/workspaces/.tokenomics-experiment-cache/<source SHA>`. Every model is checked
against its recorded SHA-256. Model files, credentials, build trees and caches
are excluded from Git.

## Workflows and evidence

| Workflow | Contract |
|---|---|
| `codespace-fixtures.yml` | Trained correctness gate through the owned lifecycle |
| `codespace-experiment.yml` | Fixed timing, pilot, confirmation and experimental CPU-profile stages; see [experiment protocol](experiment.md) |
| `profile-preflight.yml` | Both portable CPU builds for the prepared comparison branch; no trained models |
| `verify-evidence.yml` | Read-only verification of explicitly frozen run IDs and source SHAs |
| `publish-evidence.yml` | Short hosted publisher: reverify, retain immutable bytes, run the staging guard, then commit only verified results |

The artifact verifier checks the GitHub ZIP digest, inner archive digest, every
manifest member, ownership, complete block hashes, exact output IDs/stops,
cycle/cache accounting and applicable frozen-grid statistics. Archive paths,
symlinks, duplicate members and size limits fail closed. Tests cover corruption
and immutable retention.

Hosted job-log access can differ from the connected GitHub reader. In that case,
retain the exact controller report obtained through the connected job-log API
as `lifecycle.json`, with its job ID and URL. Verification checks that report
against the successful finalizer and exact run/attempt/source. A successful
inference workload alone does not establish shutdown.

Verified raw files live under `results/verification/<point>/attempt-N/`.
`verification.json` is the canonical receipt, and `lifecycle.json` records the
shutdown evidence. Rechecking preserves the first canonical receipt and refuses
changed raw bytes. Raw artifacts remain separate from derived summaries.

## Monitoring and recovery

```bash
gh run list --repo VictorWong123/tokenomics --workflow codespace-experiment.yml
gh run view RUN_ID --repo VictorWong123/tokenomics --json headSha,status,conclusion,jobs
```

Require workload success, valid artifact evidence and actual shutdown. Inspect
`TOKENOMICS_EXPERIMENT_SUMMARY` and `TOKENOMICS_CONTROLLER_REPORT`; the latter's
`fixture_evidence_valid: false` is normal for non-fixture workloads when
`shutdown_confirmed: true` is present.

For a planned pause, rerun **all jobs of the same run**, preserving its source:

```bash
gh run rerun RUN_ID --repo VictorWong123/tokenomics
```

The initializer and its dependent jobs must run together. Do not rerun only an
inference workload after its ownership lease has been finalized. Source, binary,
models, prompts, settings and order must match before cached blocks can resume.
A completed point stays immutable; changed execution code requires a new point.

If a Codex Cloud shell reports `No usable execution environment`, check the
workspace attachment independently of Codespaces. A refreshed workspace context
restored the shell during this continuation. Preserve uncommitted files before
syncing `origin/main`. If the Cloud CLI credential expires, connected GitHub
tools can still inspect runs and artifacts; keep credentials out of chat/logs.

If Codespaces startup fails before inference, retain that failure and use the
same owned initialization/dependency route for a retry. Its ownership checks
decide whether cleanup/start is permitted; do not bypass them with direct SSH
or copied lifecycle credentials.


Future development workflows retain the hosted finalizer's small `report.json`
as a separate `tokenomics-lifecycle-<run>-<attempt>` artifact. It contains only
the fixed ownership/result fields already printed by the controller, without
credentials. The verifier checks its ZIP digest, single bounded JSON member,
mode/run/attempt/source and actual Shutdown flag, and also requires the finalizer
job to succeed. Its verified controller report is also retained inside the
verification receipt, so the proof survives artifact expiry. This supports
independent shutdown checks when a hosted read
token cannot fetch another job's logs. Legacy exact-source runs retain their
job-log/API proof path, with separate lifecycle files for each attempt.

## Explicit test tags when CLI API authentication is unavailable

The existing owned experiment workflow also accepts two explicit tag namespaces:
`tokenomics-profiles-*` runs the fixed 54-record CPU comparison, and
`tokenomics-controller-packed-*` runs the separately calibrated packed controller
study. Normal code/documentation pushes do not launch these experiments. Create
a unique lightweight tag at the checked source after pushing it to main, then
push that tag. Both routes preserve the same runner ownership, global
concurrency, fixed models/settings, artifacts and always-finalize shutdown.

Before a tagged study can initialize Codespaces, a hosted preflight checks the
protocol's frozen trained-proof digest, complete 272-comparison status,
independent output checks and actual Shutdown flag. A pending/changed proof fails
before power-on. Workflow-dispatch commands remain supported when CLI API
credentials are available; tag execution does not need that shell API token.

If the shell API token expires, Git transport, connected GitHub readers and authenticated HTTP requests through
the inherited network configuration may remain usable independently. Check their
actual outcomes. An authentication failure in a local status monitor does not
establish that the inference workload failed. Do not copy lifecycle credentials
into the Cloud workspace or change the Codespaces lease to repair a monitor.

Automatic trained verification on main is limited to runtime/build/model/gate
and lifecycle sources. Artifact, benchmark-analysis and documentation publication
alone does not rerun the expensive model gate. The automatic pure C++ policy CI
remains separate; explicit workflow dispatch and checked test tags still control
measurements. Review merge-commit messages and workflow triggers before pushing
published evidence, particularly when a trained experiment is already queued.

During this continuation, an evidence merge without a skip marker launched a
redundant automatic verification. It was cancelled, its owned finalizer proved
actual Shutdown, and the paired experiment then initialized successfully. The
[operational record](../results/verification/duplicate-verification-cancel-20261108)
retains that interruption; it is not a performance failure or result.
