# Cloud kernel configuration study — diagnostic only

This uses locally manufactured transformer weights, not a trained language
model. It is a component/configuration investigation on the Codex Cloud CPU,
separate from Codespace feasibility. No new backend profile is adopted.

A subsequent packed-row prototype improves both ordinary decode and verification
in this same controlled fixture. Its separate process sequence is
guarded/packed/packed/guarded: 144 checked requests, with target-only medians
30.054/25.409 ms and cap8 medians 25.838/19.533 ms (guarded/packed).
`packed-prototype.json` records its raw hashes, sources and provenance limits.
These are synthetic component measurements, not trained-model results. The
integrated opt-in profile passes all four Release suites and all four
address/undefined-sanitizer suites with leak detection; the trained gate is
still required. It is not enabled by default.

The existing guarded profile uses consistent dot kernels for verification
widths up to nine. The alternative keeps repacking, llamafile and BLAS off but
forces the pinned tiled implementation for every supported row count. CMake
builds a generated copy; the llama.cpp submodule stays unchanged. Both profiles
pass all four Release suites, including 468 differential cases and exact logit
checks. The default also passes all four address/undefined-sanitizer suites with
leak detection (225.51 seconds).

The controlled Q4_K transformer has two layers and width 256. A period-nine
prefix makes actual upstream ngram lookup propose correct continuations at
every cap. There is no injected/oracle proposal in the final study. Each request
must match both the ordinary reference and the declared cycle exactly. All
proposals are accepted. Six rotating-order repetitions per action, repeated
in guarded/tiled/tiled/guarded process order, produce 144 checked requests.

| Profile/action | Median decode ms, 256 fixture tokens |
|---|---:|
| Guarded target-only | 30.197 |
| Guarded cap8 | 26.118 |
| All-tiled target-only | 158.740 |
| All-tiled cap8 | 26.602 |

Forcing tiling makes this target-only case 5.26 times slower. Its apparent
cap8 ratio to its own slower baseline is therefore misleading as an optimization
of the existing system; cap8 itself is slightly slower than guarded cap8.
These tiny controlled weights, cache locality and Cloud CPU are not a predictor
of Qwen, Codespace or Metal throughput. There is no trained-model speedup claim.

The new random K-quant fixture widens attention/output projections from 256 to
512 elements and exercises larger cache state. The earlier fixture already
had multiple quantization blocks in its FFN-down projection. All 468 default differential cases,
30 extra minimal/checkpoint comparisons and 18 Unicode cases pass. A separate
fresh-prefix diagnostic retains raw wide-fixture logits for both profiles.
In that one case the greedy winner is unchanged; the second-best logit differs
by one ULP across profiles. Raw differences alone do not prove an emitted-token
mismatch, and this first-row diagnostic is not a complete greedy gate or an
original-history replay experiment.

An initial study harness tried to fabricate proposals after an empty upstream
draft. Upstream correctly asserted its acceptance-state contract. That invalid
attempt is excluded; its failure is retained in `invalid-oracle.stderr.txt`.
The final program uses genuine ngram lookup with no callbacks. The test-only
fault hook is suitable for corrupting existing drafts, not inventing a draft
without an active upstream implementation.

`summary.json` retains the hardware snapshot, exact source/weight/binary hashes,
selected flags, all action medians and limitations. Its hardware snapshot was
collected after the experiment; live load/pressure is not start telemetry.
`raw/` contains every accepted timing record and the separate diagnostic jobs,
requests and cycles. Source is in `source/`. No GGUF weights are committed.

To reproduce, use the repository virtual environment and pinned submodule:

```bash
.venv/bin/python - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, 'tests')
from make_fixtures import create
p = Path('results/local/kernel-study-repro/repeat-k.gguf')
p.parent.mkdir(parents=True, exist_ok=True)
create(p, 259, k_quantized=True, ngram_cycle=True)
PY
kernel_study_source="$PWD/results/verification/kernel-study-20261108/source"
kernel_study_args=(-G Ninja -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_MAKE_PROGRAM="$PWD/.venv/bin/ninja"
  -DPython3_EXECUTABLE="$PWD/.venv/bin/python"
  -DGGML_NATIVE=OFF -DGGML_METAL=OFF -DGGML_CUDA=OFF
  -DGGML_CPU_REPACK=OFF -DGGML_LLAMAFILE=OFF -DGGML_BLAS=OFF
  "-DCMAKE_PROJECT_tokenomics_INCLUDE=$kernel_study_source/tokenomics_tiled_study.cmake")
.venv/bin/cmake -S . -B build-study-guarded "${kernel_study_args[@]}" \
  -DTOKENOMICS_CPU_BATCH_INVARIANT=ON -DTOKENOMICS_STUDY_FORCE_TILE=OFF
.venv/bin/cmake -S . -B build-study-tiled "${kernel_study_args[@]}" \
  -DTOKENOMICS_CPU_BATCH_INVARIANT=OFF -DTOKENOMICS_STUDY_FORCE_TILE=ON
.venv/bin/cmake --build build-study-guarded --target tokenomics_kernel_study -j4
.venv/bin/cmake --build build-study-tiled --target tokenomics_kernel_study -j4
build-study-guarded/tokenomics_kernel_study guarded results/local/kernel-study-repro/repeat-k.gguf 1
build-study-tiled/tokenomics_kernel_study tiled_all results/local/kernel-study-repro/repeat-k.gguf 1
```

Finish builds before measurements and run the two processes serially. Repeat
the last two commands in reverse order to reproduce the original process order.
