# Tokenomics

A C++20 speculative-decoding runtime with a cost-aware controller, built around
a pinned [llama.cpp](https://github.com/ggml-org/llama.cpp). The native executable
owns generation, policy decisions, retained KV caches, rollback, lazy draft
synchronization and target-only fallback. Upstream supplies model execution,
draft generation and greedy verification.

Python supports offline calibration, tests, benchmark orchestration and analysis.
The model and online controller execute in C/C++; Python is not in the token
generation loop. See the [source and language map](docs/development.md#source-and-language-map).

## Build the C++ executable

Requires Git, a C++20 compiler, CMake 3.24+ and Ninja. This build does not require
Python, downloaded models or a GPU:

```bash
git clone --recurse-submodules https://github.com/VictorWong123/tokenomics.git
cd tokenomics
cmake -S . -B build-native -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=OFF -DGGML_NATIVE=OFF -DGGML_METAL=OFF -DGGML_CUDA=OFF
cmake --build build-native --target tokenomics -j2
build-native/tokenomics --help
```

Run with an existing compatible GGUF target:

```bash
build-native/tokenomics --target /path/to/target.gguf \
  --policy target --max-tokens 128 --prompt 'Explain binary search.'
```

For `cost-aware`, supply measured frozen priors with `--controller FILE.json`.
The [controller guide](docs/controller-design.md) documents calibration, actions,
transition scoring and supported CLI combinations. Model download sources and
licenses are recorded in [the model catalog](configs/models-v1.json).

## Development and tests

The full development environment additionally needs Python 3.12+, venv support
and curl. The bootstrap installs pinned tooling, builds the runtime and runs
all six CTest suites:

```bash
bash scripts/bootstrap.sh cpu
# macOS with Xcode Command Line Tools: bash scripts/bootstrap.sh metal
.venv/bin/ctest --test-dir build-cpu --output-on-failure
```

Set `TOKENOMICS_BUILD_JOBS=2` on a memory-limited host. The
[development guide](docs/development.md) covers C++-only policy tests, full
integration tests, sanitizers and formatting. A physical Apple M3 also passed
the Metal fixture pipeline and a trained ngram controller check. See the
[Mac and Linux comparison](results/verification/mac-m3-metal-20261111/README.md).

## Inspect the measured controller

Replay audited trained-model decisions without downloading models or running
inference:

```bash
.venv/bin/python scripts/demo_controller.py --max-cycles 12
```

The completed Linux CPU development study contains 252 exact validation records
on six prompts, two output caps and three repetitions. Controller throughput
relative to each named baseline is:

| Baseline | Controller / baseline |
|---|---:|
| Ordinary target-only decoding | 0.9709× |
| Development-frozen fixed1 | 0.9943× |
| Existing ngram cap8 | 1.2048× |

Decision/update work averages 1.67 µs per cycle. These results establish no
ordinary-decoding acceleration. See [results and uncertainty](docs/controller-results.md),
the [four-page report](results/verification/engineering-report-20261108/tokenomics-engineering-report.pdf)
and the [retained evidence index](results/README.md).

The experimental packed CPU profile passes its within-profile correctness gate
but fails cross-profile greedy equality. Guarded CPU remains the default.
Full Pilot A/B and the 90-prompt held-out evaluation remain incomplete. See
[requirements and status](docs/requirements.md).

## Repository map

| Directory | Responsibility |
|---|---|
| `src/` | Owned C++ runtime, policies, commit rules, CLI and configuration |
| `tests/` | C++ correctness tests, fixture generation and Python integration tests |
| `bench/` | Offline calibration, frozen workloads and resumable benchmark runner |
| `analysis/` | Statistics, plots and report generation |
| `scripts/` | Documented build, experiment, diagnostic and operational entry points |
| `configs/` | Versioned protocols and model catalog |
| `docs/` | Design, development, execution and measured conclusions |
| `results/` | Immutable evidence and derived reports; local runs are ignored |
| `third_party/llama.cpp/` | Clean, pinned upstream submodule |

Start with the [documentation index](docs/README.md),
[benchmark commands](docs/benchmarking.md) or [Python script catalog](scripts/README.md).
Develop in Codex Cloud; use the existing Codespace for trained execution through
the [owned lifecycle](docs/environments.md) after checking and pushing source.

## Scope and licenses

The runtime supports greedy decoding and full-attention models with partial KV
removal. Stochastic sampling, context shifting and hybrid/recurrent memory are
outside the verified contract. Authoritative output is the token-ID sequence;
budget-truncated UTF-8 display text may contain a replacement character.

Owned software is MIT and original prompts are CC0. Upstream software and model
licenses retain their own terms; Qwen2.5-3B has Qwen Research terms. No model
weights are committed. [Ownership and upstream reuse](docs/upstream.md) explains
the exact contribution.
