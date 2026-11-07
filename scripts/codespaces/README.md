# Hybrid dedicated Codespace fixtures

Private trusted-writer infrastructure. Pushes to canonical `main` and explicit
manual **Run workflow** actions trigger Linux verification. There are no PR or
scheduled triggers. Global concurrency serializes runs, without cancellation.
The legacy portability workflow is manual-only; its optional Linux/macOS
commands are preserved. The current fixed workload runs trained-model
diagnostics and the 272-comparison correctness gate; see
`docs/model-correctness.md`. The fixture contract below also remains supported
by the lifecycle controller.

Failed initialization, workload or finalization runs a short hosted reporter
after cleanup, with `issues: write` on its ephemeral Actions token. It creates
or updates an issue per failure signature, retaining exact token mismatches,
commit/run links and whether shutdown was confirmed. The Codespace workload
has no issue-write or lifecycle-token access. Cancelled tests are preserved as
cancelled runs rather than automatically retried or filed as failing cases.

Short hosted `initialize` and `finalize` jobs perform lifecycle work (normally
about one minute each; five-minute ceilings). The long workload uses the
repo-only self-hosted label `tokenomics-codespace-long-tasks`, with a 210-minute
ceiling including setup and artifact transfer. Self-hosted Actions execution
uses no hosted runner minutes; Codespaces compute/storage remains billable.
Do not use an expensive hosted runner to wait for the workload. The fixed CPU
fixture subprocess has a 200-minute deadline, build parallelism two, and
`--threads 2`; current fixture code overrides inference threads to one.

Prerequisites, provisioned once by the runtime owner:

- A **new** repo-only runner registration, never copied old ID 2 credentials.
  Its exact name/ID are repository variables `TOKENOMICS_RUNNER_NAME` and
  `TOKENOMICS_RUNNER_ID`. Register it under
  `/workspaces/.tokenomics-actions-runner`, current UID, directory mode 0700,
  with repository URL exactly `https://github.com/VictorWong123/tokenomics` and
  label `tokenomics-codespace-long-tasks`. No auto-start listener/service/tmux.
- Repository variable `TOKENOMICS_CODESPACE`: dedicated private Tokenomics
  two-CPU machine, **Shutdown**, created with idle timeout **240 minutes**.
  The controller validates 240; it never PATCHes idle timeout. The platform
  maximum is 12 hours; this workflow does not request it.
- Repository secret `CODESPACES_TOKEN`: fine-grained, only Tokenomics repository,
  Codespaces read and Codespaces lifecycle admin write. Only hosted lifecycle
  steps receive it. Never enter it in Cloud or the Codespace. The workload sees
  only normal ephemeral Actions job/runtime credentials used by checkout and
  artifact upload; no lifecycle PAT is sent over SSH or installed remotely.
- `/usr/bin/python3` 3.12 with venv/ensurepip; bash, git, tmux, GNU timeout/stat,
  C/C++20 compiler/build tools and system Ninja discoverable on PATH.
  The fixture PATH is `/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin`, ensuring
  system Python wins while allowing `/usr/local/bin/git`. Public pinned
  llama.cpp/submodule and Python package-index network access are required.
  Bootstrap installs the pinned CMake/Ninja/Python requirements, builds CPU,
  and runs four CTest tests. No real model download or project pilot is run.

Initialize admits only exact target metadata and Shutdown. SSH wakes it and
atomically creates `/dev/shm/tokenomics-codespace-lifecycle.lease` after verifying
tmpfs. Exclusive/no-follow creation, mode 0600, current UID and regular-file
checks protect the exact SHA/run/attempt identity. A persistent exact operation
marker lives at `/workspaces/.tokenomics-operations/<run>-<attempt>/owner.json`.
The helper validates BOM-tolerant `.runner` metadata and refuses any existing
worker/listener/tmux conflict, then starts only the fixed tmux listener session
and confirms `Listening for Jobs` from its freshly truncated per-run-owned log
plus the current exact listener process. Old runner logs are never readiness evidence. Only then is `claimed=true` published.
Failed readiness attempts owner-verified cleanup; a nonowner never stops.
If lease acquisition/transport fails without provable ownership, initialization
fails closed and may leave a woken machine for operator inspection/idle fallback.

The workload starts with exact lease/marker/registration verification, and uses
its own clean Actions workspace at the exact triggering SHA, never the original
`/workspaces/tokenomics` checkout. Checkout credentials are not persisted. Its
results directory is unique to the owned run; existing results are rejected.
The genuine producer contract is 24 blocks and 168 request/correctness records
(seven configurations per block), every correctness `ok` true. Aggregate
`cycles.jsonl` does not exist; cycles remain in per-block raw output.

Canonical evidence is a bounded archive containing root `status.json` and
`environment.json`, fixture-pipeline `complete.json`, `correctness.jsonl`,
`requests.jsonl`, `run_manifest.json`, and a nonsecret exact execution manifest.
Member names/types/sizes, JSON/count/block layout, correctness and identity are
validated. The artifact also carries the fixed nonsecret controller report,
with three-day retention, never runner logs/registration files, credentials,
build/venv trees, real models or broad results. A failed archive is not uploaded.
Only a strict compact summary with source/host, 24/168 counts and fixed file/
archive digests is passed to finalize; prompts/raw records/hardware are omitted.

Finalize runs with `always()` after failed/cancelled workloads when initialize
claimed ownership. An already Shutdown machine is confirmed without SSH.
Otherwise only Available is accepted: verify lease/marker, wait for no worker,
stop the fixed owned tmux listener, remove only the owned operation root, retain
the tmpfs lease, then request stop and poll actual Shutdown. Ownership ambiguity,
active workers or failed cleanup prevents stopping a possibly foreign session.
Stop failure fails closed. The lease fences new controllers until Shutdown;
runtime owner has demonstrated tmpfs clears on stop/start, but stale leases
are never deleted automatically. Rerun all jobs together, not workload alone.

Only actual Shutdown plus a schema/identity-valid workload summary permits one
`TOKENOMICS_EVIDENCE_SUMMARY` log line. Failure/cancellation still attempts stop. Missing/malformed/custom workload
evidence produces only a fixed nonsecret lifecycle report with confirmed
shutdown and `fixture_evidence_valid=false`, never a fixture success line.
Cleanup success does not turn a failed workload job into a passing workload. Canonical artifacts remain on
Actions. Hard hosted outage/cancellation or lost runner connectivity can prevent
cleanup despite `always()`; inspect the machine and rely on the creation-time
240-minute idle fallback as a backstop, not a precise maximum billing guarantee.
Offline tests: `python3 -m unittest discover -s scripts/codespaces -p 'test_*.py'`.

## Reusing the remote computer for authorized long tasks

For a future approved roughly three-hour task, a Cloud author may change the
trusted self-hosted workload step and its checked-in helper/result allowlist to
that task's fixed command and explicit expected output paths. Keep the hosted
initialize/finalize jobs, exact lease/marker checks, runner variables/label,
secret isolation and `always()` finalize dependency. Review source before a
trusted push; there is no arbitrary command/input interface. The current
workload wrapper is 200 minutes (not the retired 20-minute wrapper), within the
210-minute job budget and 240-minute creation-time idle setting. Size any future
fixed task and evidence transfer to fit those budgets. Current 24/168 fixture
validation is local to fixture evidence generation; it cannot gate generic
ownership cleanup or shutdown. Custom tasks without fixture summaries still
get cleanup-only finalization, never `TOKENOMICS_EVIDENCE_SUMMARY`. Define an
appropriate separate success/result contract for a future custom task rather
than pretending its output is fixture evidence.
