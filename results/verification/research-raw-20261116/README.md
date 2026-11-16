# Preserved public CPU research payloads

These are the original four GitHub artifact ZIPs, not reconstructed transcripts.
Each companion JSON binds the public run/source, artifact ID, GitHub ZIP digest,
byte count and all internal completion seals. The recovery workflow
[43865219412](https://github.com/VictorWong123/tokenomics/actions/runs/43865219412),
source `f8a980a9c14a30237d793ea9e2a603cc86e3fe7d`, downloads only these allowlisted
public diagnostic artifacts on a standard Ubuntu runner. No inference, Codespace,
weights, executable, credentials or environment dump is involved.

The archive SHA256s and every sealed payload hash were rechecked after log
transport. `offline-audit.json` additionally rechecks440 actual learned-model
comparisons and15955 cycle records: exact prompt/token IDs and stops, per-cycle
commits/work, direct-sampler eligibility/calls, equal sampler work, and guided
work/hint ages. The guided archives contain original rank collections and live
request/cycle payloads, including the actual NEON/SSE2 markers. Nothing depends
on the seven-day artifact retention period now.

| Run | Scope | Comparisons | Cycles |
|---|---|---:|---:|
| 43848978979 | 0.5B direct sampler, Apple ARM CPU | 160 | 4638 |
| 43849593799 | 1.5B direct sampler, Apple ARM CPU | 160 | 4934 |
| 43861391090 | 0.5B guided live validation, Apple ARM CPU/NEON | 60 | 2994 |
| 43861395334 | 1.5B guided live validation, Linux CPU/SSE2 | 60 | 3389 |

These are exactness/work checks, not M3/Metal throughput measurements. Timings in
raw payloads are hosted diagnostic observations and are not treated as performance
results. Models and source revisions differ as recorded; do not pool trajectories.

```bash
.venv/bin/python -m analysis.research_archive \
  results/verification/research-raw-20261116 \
  --out results/local/research-archive-audit.json
```

This audit detects accidental alteration; it cannot authenticate an archive and
all its separately recorded digests if an adversary replaces them together.
