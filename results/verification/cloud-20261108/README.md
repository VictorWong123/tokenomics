# Cloud implementation verification

Source `87c7811ed5fe24024f50782a1307ecbcef52bdee`, Release CPU, pinned llama.cpp.
This is Codex Cloud evidence, separate from Codespace trained-model evidence.

- 330 transformer fixture cases plus 25 minimal/checkpoint comparisons.
- All four CTest suites passed in Release and with address/undefined sanitizers
  and `ASAN_OPTIONS=detect_leaks=1` (96.16 seconds for sanitizer CTest).
- 24 separate lifecycle tests passed.
- Complete fixture pipeline: 24 blocks, 168 exact-ID request checks, checkpoint
  replay and transition tests. Example analysis and terminal demo executed.

The fixture pipeline ran in `results/local/verified-87c7811`. Its frozen code
fingerprint, binary hash and hardware are summarized in `verification.json`.
Synthetic weights and their timings are not language-model feasibility evidence.

Reproduce portable checks with `bash scripts/bootstrap.sh cpu`, followed by
`bash run.sh cpu --fixtures-only --run-dir results/local/my-fixtures`.
For sanitizer checks use a separate build directory with
`-DTOKENOMICS_SANITIZE=ON -DCMAKE_BUILD_TYPE=Debug`, then run CTest with leak
detection. The README's bootstrap supplies the virtualenv CMake/Ninja paths.
