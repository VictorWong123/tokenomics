# M3 research handoff — 16 November 2026

Use branch `research/speculative-throughput-20261116`. Main and the pinned
llama.cpp submodule are unchanged. No Codespace is needed. The next decisive
result is actual M3 Metal verification and paired generation timing; hosted
Apple virtual machines compile the shaders but lack the required GPU SIMDgroup
features. CPU proof and component timings do not establish an M3 win.

## Prepare the Mac

Use native arm64 Python 3.12+, Xcode Command Line Tools, AC power and a quiet machine.
Fetch and switch to the research branch without discarding existing local work:

```bash
git fetch origin
git switch research/speculative-throughput-20261116
git pull --ff-only
git submodule update --init --recursive
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

The default target is the pinned 0.5B model, 491400032 bytes, SHA256
`74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db`.
Reuse the existing matching file, or download it explicitly:

```bash
.venv/bin/python scripts/download_models.py --model qwen25-05 \
  --lock configs/metal-model-header-pins-v1.json --directory models
```

Do not run studies concurrently. Each command needs a fresh output directory;
failed attempts are retained. Leave source and binaries unchanged during a run.

## First: inspect the Metal matrix candidates without a model

```bash
.venv/bin/python scripts/metal_matvec_study.py \
  --out results/local/m3-components-01
```

This builds plain short-MV, interleaved, grouped-Q8 and shared-Q8 component
executables. It checks bitwise row equality before measuring each shape, retains
all 6400 timing pairs and 400 gates, and rejects a CPU fallback. The large Q8
shape matches the 0.5B output head. Cache-warm matrix ratios exclude attention,
KV, sampling and complete model traffic. See [component protocol](metal-matvec-component.md).

## Primary full-model candidate: shared Q8 verification

```bash
.venv/bin/python scripts/metal_short_mv_study.py --share-q8 --screen retrieval \
  --model models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --out results/local/m3-q8-shared-01
```

The runner builds isolated reference/candidate binaries, executes the actual
GPU backend-row suite, then 720-request trained gates for ngram and suffix.
Only passing exact IDs/stops, cache/work and complete 25/25 GPU-offload checks
enable the 126-record paired development screen. The screen includes unchanged
ordinary decoding, candidate ordinary decoding, and fixed ngram/suffix 1/4/8
controls. A faster candidate must beat the appropriate ordinary and strong
fixed-policy controls; a kernel component improvement alone is insufficient.

A failing gate is useful evidence: preserve the entire fresh directory, including
the first mismatch. Do not enable timings by bypassing it. The interleaved
candidate is a separate alternative, using `--interleave` instead of
`--share-q8` and a different output directory. Plain short-MV omits both flags.
These experiments remain OFF in ordinary builds. See [Metal design/gates](metal-short-mv.md).

## Separate CPU sampling improvement

```bash
.venv/bin/python scripts/metal_short_mv_study.py --direct-greedy --screen retrieval \
  --model models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --out results/local/m3-direct-greedy-01
```

This preserves the guarded one-row Metal verifier and isolates strict host
greedy sampling. It avoids redundant vocabulary conversion/filter scans; it can
benefit ordinary decoding too, so optimized ordinary controls are essential.
Both 0.5B and 1.5B learned Apple ARM CPU gates pass 160 comparisons each with equal
work and actual eligibility/call checks. This is still not measured M3 Metal
speed. Do not combine this flag with the matrix experiments in the first study.
See [sampling implementation](greedy-sampler.md).

## Optional larger target

The pinned 1.5B model is 1117320736 bytes, SHA256
`6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e`.
It is supported by the local study/gate with 29/29 required GPU-offloaded layers.
The no-model diagnostic also accepts `--model-id qwen25-15`, comparing plain
short-MV/interleaving at five actual projection shape/type classes before loading weights.
Its tensors use F32/Q4_K/Q6_K and contain no Q8; shared/grouped-Q8 flags are
therefore rejected. Plain short-MV, interleaving or direct sampling apply:

```bash
.venv/bin/python scripts/download_models.py --model qwen25-15 \
  --lock configs/metal-model-header-pins-v1.json --directory models
.venv/bin/python scripts/metal_short_mv_study.py --model-id qwen25-15 \
  --interleave --screen retrieval \
  --model models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
  --out results/local/m3-15-interleaved-01
```

Run this after the 0.5B diagnostic to avoid spending memory/time on an unresolved
backend issue. The study records memory and thermal conditions before/after.

## Results already available

- Native 1072-case backend harness: 7508672 bitwise CPU comparisons, plus native
  Apple host/shader compilation. The virtual GPU cannot execute the Metal gate.
- Default-OFF strict direct sampler: exhaustive first-index/NaN/tie tests and 320
  learned ARM comparisons; backend GGML argmax fails the required tie contract.
- Default-OFF logit-guided copy: 120 learned CPU comparisons. Together with the
  sampler runs, all 440 comparisons and 15955 raw cycles are independently audited.
  Guided caps 4/8 are dominated by suffix on 0.5B CPU validation; 1.5B gives a work tradeoff
  against ngram and adds ranking overhead. It is not a deployment recommendation.
- Linear transition/stale-logit follow-up: 336 exploratory per-prompt records,
  17416 computed cycles, too much wasted verification work; no native drafter added.
- Suffix-priority combinations: 384 post-hoc records and 19616 computed cycles;
  guidance gives work tradeoffs and ngram fallback mostly reproduces ngram.
- Reserved 90-prompt held-out evaluation remains untouched. No new candidate has
  established an M3 throughput win in this session.

[Retained evidence](../results/README.md), [session research record](research-session-20261116.md)
and [paper-based priorities](metal-research-directions-20261116.md) provide the
full scope, failed hypotheses and reproduction commands.

To audit the retained CPU proposal/sampler/census results without downloading
models or executing inference:

```bash
.venv/bin/python scripts/audit_research_session.py \
  --out results/local/research-audit-01
```

This verifies raw learned records and seals, recomputes all 16 proposal collections
and their per-cycle proposals/work/hints, and reconciles the projection census.
It does not execute Metal or replace the local gates above. If a new kernel
profile wins the development screen, freeze it and collect fresh calibration
before using the cost controller; its old serial-verifier priors are unsuitable.
