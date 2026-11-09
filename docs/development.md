# Development guide

## Source and language map

The deployable program is `tokenomics`, a C++20 executable. GGUF files contain
model weights; llama.cpp/ggml executes their native computation. Python launches
whole requests and analyzes recorded results. It does not select actions or
compute neural-network layers in the generation loop.

| Owned C++ source | Responsibility |
|---|---|
| `src/runtime.cpp`, `runtime.h` | Engine/model lifecycle, generation, KV retention/rollback, synchronization, snapshots and stage timers |
| `src/policy.cpp`, `policy.h` | Online action scores, EWMA, probes, hysteresis, fallback and reset; fixed/acceptance/script policies |
| `src/commit.cpp`, `commit.h` | Draft budgets and accepted/correction/bonus token commitment |
| `src/controller_io.cpp`, `controller_io.h` | Strict controller JSON and trace serialization |
| `src/main.cpp` | CLI, JSONL jobs/results and request metadata |
| `tests/test_*.cpp` | Native policy, configuration, commit and differential runtime tests |

`third_party/llama.cpp` supplies tokenization, model execution, drafting and greedy
verification. CMake generates guarded/packed native source copies in the build
tree; the pinned submodule stays clean. See [upstream](upstream.md) and
[CPU profiles](cpu-profiles.md) before changing those paths.

| Python area | Responsibility and reason |
|---|---|
| `bench/` | Workload construction, request ordering/resume, metadata and offline calibration; JSON/process orchestration |
| `analysis/` | NumPy statistics, Matplotlib figures and PDF generation; shared paired statistics live in `analysis/report.py` |
| `scripts/` | Model download, study/evaluation entry points, diagnostics, demos and evidence audits; see the [catalog](../scripts/README.md) |
| `scripts/codespaces/` | Runner/start/stop ownership and operational tests |
| Python files in `tests/` | Tiny GGUF generation and CLI, calibration, benchmark and artifact-contract tests |

`bench/controller.py` computes **offline initial priors**; online adaptation is
C++. `scripts/codespaces/controller.py` manages the machine/runner, not inference.
CMake, shell and workflow YAML handle build/automation.

## Native build and C++-only tests

Requires a C++20 compiler, Git, CMake 3.24+ and Ninja. Initialize the exact clean
submodule with `git submodule update --init --recursive`.

```bash
cmake -S . -B build-native -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=OFF -DGGML_METAL=OFF -DGGML_CUDA=OFF \
  -DBUILD_TESTING=ON -DTOKENOMICS_PYTHON_TESTS=OFF
cmake --build build-native --target tokenomics test_core test_controller test_controller_io -j2
ctest --test-dir build-native --output-on-failure
build-native/tokenomics --help
```

This registers the three pure C++ suites: `core`, `controller` and
`controller_io`. It does not discover Python. `test_runtime` remains a buildable
C++ target, but its automatic fixture setup and CTest registration require full
development mode. A native-only pass is not a runtime/model gate.

Use `-DBUILD_TESTING=OFF` for just the executable/libraries. On macOS select
`-DGGML_METAL=ON`; CUDA uses `-DGGML_CUDA=ON` and its native toolchain. These
options do not establish hardware correctness or performance by themselves.

## Full development environment

Additionally requires Python 3.12+, venv support and curl. macOS needs Xcode
Command Line Tools; Linux needs its C++ toolchain and Python venv OS package.

```bash
bash scripts/bootstrap.sh cpu
# bash scripts/bootstrap.sh metal   # macOS only
```

Bootstrap initializes the submodule, creates `.venv`, installs pinned CMake,
Ninja and Python analysis/test dependencies, builds `build-cpu` and runs CTest.
Set `TOKENOMICS_BUILD_JOBS=2` to limit compilation parallelism. Python packages
are development/experiment dependencies, not native runtime dependencies.

`TOKENOMICS_PYTHON_TESTS` defaults to `ON`, preserving the six-suite contract.
In a reused build directory, enable it explicitly if it was disabled:

```bash
.venv/bin/cmake -S . -B build-cpu \
  -DTOKENOMICS_PYTHON_TESTS=ON -DPython3_EXECUTABLE="$PWD/.venv/bin/python"
.venv/bin/cmake --build build-cpu -j2
.venv/bin/ctest --test-dir build-cpu --output-on-failure
```

| Suite | Coverage |
|---|---|
| `core` | C++ commit/budget and baseline policy rules |
| `controller` | C++ scores, EWMA, probes, holds, transitions, reset and allocation-free valid decisions/updates |
| `controller_io` | C++ strict configuration and trace JSON |
| `fixtures` | Python-generated deterministic/random and quantized tiny GGUFs |
| `runtime` | C++ transformer execution and differential IDs/cache/stopping checks against fixtures |
| `tooling` | Python CLI, calibration, resume, analysis and artifact contracts, including fixture-backed executable calls |

Synthetic fixtures establish correctness, not trained-model throughput. Trained
checks and measurements follow [benchmarking](benchmarking.md) and the owned
[Codespaces lifecycle](environments.md).

## Test organization

Python discovery remains `python -m unittest discover -s tests -p 'test_*.py'`.
CTest supplies `TOKENOMICS_TEST_BINARY` and creates fixtures. Direct discovery
without that variable skips executable-dependent cases.

The former broad `test_tooling.py` is split into `test_benchmark_runner.py`,
`test_analysis.py`, `test_confirmation.py`, `test_model_gate.py`,
`test_run_pipeline.py` and `test_artifact_io.py`. Every original test method
remains. Controller calibration/CLI/profile and artifact/diagnosis tests retain
their focused modules. Run the separate lifecycle suite with:

```bash
.venv/bin/python -m unittest discover -s scripts/codespaces -p 'test_*.py'
```

Optional two-profile tests also need `TOKENOMICS_PACKED_TEST_BINARY` pointing to
a separately checked packed build. Skipped cases are not profile validation.

## Sanitizers and formatting

```bash
.venv/bin/cmake -S . -B build-asan -G Ninja \
  -DCMAKE_MAKE_PROGRAM="$PWD/.venv/bin/ninja" \
  -DPython3_EXECUTABLE="$PWD/.venv/bin/python" \
  -DCMAKE_BUILD_TYPE=Debug -DTOKENOMICS_SANITIZE=ON \
  -DGGML_NATIVE=OFF -DGGML_METAL=OFF -DGGML_CUDA=OFF
.venv/bin/cmake --build build-asan -j2
ASAN_OPTIONS=detect_leaks=1 .venv/bin/ctest --test-dir build-asan --output-on-failure

.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/black --check bench analysis scripts tests
.venv/bin/clang-format --dry-run --Werror src/*.cpp src/*.h tests/*.cpp
```

Run checks appropriate to a change in Codex Cloud before publishing source.
Use Codespaces for trained execution after those checks.

## Change and evidence rules

- Preserve the pinned clean upstream and its verifier. Own changes in `src/` or
  the checked CMake integration; explain the ownership boundary.
- Preserve exact IDs/stops, request reset and charged transition costs.
- Share complete-record JSONL writing through `bench.common.write_jsonl`.
  Callers own directory creation, atomic replacement and incremental streaming.
- Keep experiment entry-point paths stable: workflows and frozen evidence refer
  to them. Add shared helpers rather than breaking historical commands.
- Keep existing raw records, hash-bound configs and canonical receipts immutable.
  Use new run directories after source/binary/model/settings changes. Local
  builds/models/runs belong under ignored paths; do not merge timing points.
- Update current documentation with behavior changes. Keep chronological session
  notes separate from current build instructions and status.

Stage explicit paths and run `.venv/bin/python scripts/check_staged.py` before
committing. See the [documentation index](README.md) for scientific contracts.
