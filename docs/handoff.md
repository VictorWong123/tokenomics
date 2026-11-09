# Tokenomics continuation handoff

## Repository cleanup (9 November 2026)

The root README now leads with the C++ executable and measured scope. Build/test
instructions and the C++/Python source map are in `docs/development.md`, full
benchmark commands are in `docs/benchmarking.md`, and `scripts/README.md` catalogs
the stable tools. `docs/README.md` and `results/README.md` index guides/evidence.

`TOKENOMICS_PYTHON_TESTS=OFF` enables the native executable and three pure C++
policy/configuration suites without discovering Python. The default remains ON,
and bootstrap explicitly enables the full six-suite development contract.
Policy CI uses the native-only setting; it is not a model-correctness gate.

The former broad Python tooling test module is split by contract, retaining all
65 original discovered Python test methods. A shared JSONL writer removes repeated
complete-record encoding; an additional compatibility check preserves existing
artifact bytes. Runtime/controller C++ source and archived measurement payloads,
canonical receipts and hash-bound configurations are unchanged by this cleanup.

Cloud checks pass: a fresh native executable and all three native policy/config
suites without Python discovery; all six full-development CTest suites; all 66
Python tests with both CPU-profile binaries supplied; and all 24 lifecycle tests.
The native executable also emits 16 synthetic-fixture tokens directly. Recorded
demo/example analysis commands work, and regenerating the controller summary
matches the canonical trained result exactly. No new trained performance result
or Codespace execution follows from this cleanup.

## Current continuation (8 November 2026)

Current checkpoint:

- The experimental C++ controller is implemented, documented and pushed to main.
  Its inference source is `e98c0fa`; subsequent changes are offline tooling/docs.
- Guarded and packed Release plus ASan/UBSan/leak checks pass six suites,
  including 906 differential fixture cases. Automatic pure-policy CI also passes.
- Controller trained gates pass 28 model-drafter and 28 ngram records, covering
  every action. Raw IDs/stops/cache accounting are independently checked.
- Study 42492089188 attempt 2 completes all 36 blocks / 252 exact validation
  records and 16 logging records. Both attempts have actual Shutdown proof.
  All 1587 payload files plus the manifest pass audit; the first 25 blocks, fit and prior are
  byte-identical after resume. Canonical proof and raw attempt 2 are retained.
- Controller/target throughput is 0.9709 [0.9607,0.9810]; controller/fixed1
  0.9943 [0.9853,1.0023]; controller/cap8 1.2048 [1.1271,1.2868]. No practical
  ordinary-decoding gain. Policy decision/update averages 1.67 µs/cycle;
  verification consumes 99.992% of generation. See controller-results.md.
- Packed gate 42503429351 attempt 2, source 708993c, is complete and independently
  verified: 272 comparisons, 14 diagnosis records, 18 screen requests and 66
  original-state replay actions. All 166 payload files plus the manifest are
  retained. Finalizer 128849625747 and the small lifecycle artifact prove Shutdown.
  Binary ab5530171da6c3168e8bc40af144bc2cd5a7c3deddbfb27c5b333193ef31620f.
- Both prepared profile protocols now pin canonical gate SHA
  cac71c63a37ba0775c314314b67837cedd4d42588619b117e6b263a648bde0ad.
  The 54-record paired study ran as 42519929357, exact source
  c834026c6bdddc4d219a25945befc318361e74fb, via lightweight tag
  tokenomics-profiles-20261108-c834026c6b. It is allowlisted for independent
  verification. It stopped on the first block: cross-profile token 36 differs
  (guarded 323 / packed 882); all variants agree within their own profile. All 36
  payload files and actual Shutdown are checked. No cross-profile performance
  summary is accepted.
- Diagnosis 42523392101 reproduces both full 128-token ordinary histories and
  warm-ups at the original exact executable/model hashes. At token 36, the top
  two logits reverse: guarded 323 / 882 (gap 0.244980), packed 882 / 323 (gap 0.031298).
  All 11 payload files and the manifest pass independent audit. The subprocess
  completed before a report-writer bug failed the workflow; the receipt records
  that failure and missing report explicitly. Finalizer 128865017149 confirms
  actual Shutdown. The writer is fixed and tested; no model rerun is needed.
- Final hosted evidence check42525108573 passes from published source97f89da,
  independently rechecking all five frozen archives without inference or a
  Codespace start. Its receipts are retained in
  `results/verification/hosted-evidence-check-20261108`; the diagnostic receipt
  matches canonical, including the original workflow reporting failure.
- The separately named controller-packed extension is locally checked and now
  has a frozen prerequisite. It has no trained performance result yet. Use fresh
  calibration; run it only if the paired component result justifies that point.
- The shell GH_TOKEN returned401 at20:06; Git transport, connected GitHub tools
  and environment-authenticated Python HTTP requests still work. The latter
  successfully dispatched the diagnosis without copying a credential. The old
  monitor exited because of auth, not an inference failure. No Cloud reconnection
  is needed; do not copy the lifecycle secret into Cloud.
- Full 90-prompt held-out model evaluation is unexecuted; its guarded entry point
  and supplemental analysis are implemented/tested. The older CPU point suggests
  roughly 119 hours for 6300 conditions, not a same-day default experiment.
- Preserve source/binary/model/configuration bindings. Use attempt-specific
  lifecycle proof for the legacy controller run. Future workflows retain a small
  hosted shutdown artifact automatically. Do not start a competing Codespace job
  or apply the already-integrated named stash again.

The detailed design/API/commands are in `docs/controller-design.md`; full-grid
rules and figures are in `docs/evaluation.md`. Follow actual artifact/finalizer
status rather than inferring completion from an in-progress workload.

## Earlier decisions

The [engineering history](engineering-history.md) preserves earlier setup,
implementation, experiment and recovery notes. Use this handoff for current
status; historical queue states are not instructions to restart old work.
