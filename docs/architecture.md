# Runtime and correctness contract

One process, one active sequence, greedy decoding, F16 KV, fixed target GGUF,
batch/ubatch and thread counts. CPU is the default; Metal/CUDA are optional
build choices. "Portable" means a reproducible supported llama.cpp backend on
Linux/macOS, not a claim of tested acceleration on every computer. Windows can
use WSL2; native Windows resource measurement is not implemented.

At a boundary the target caches a committed prefix P and has one pending token L.
P + L is the exact prompt plus generated sequence so far. The final emitted
token is pending, so cache length and generated count differ by one.

1. Choose k (0 means target-only), clamp to output and context capacity reserving
   one target correction/bonus slot.
2. Before model speculation, decode missing P tokens into the draft. Its cache
   must be a prefix of P. This supports lazy fallback and re-entry.
3. Ask upstream for up to k proposals; it processes L and all but its final
   proposed token. Do not assume a fully accepted block leaves both caches equal.
4. Target-decode [L, proposals], then call upstream sample-and-accept.
5. Stop at the first target EOG or output cap; append the old pending token and
   each committed token except the new pending token to P.
6. Remove target/draft KV after new P. Assert the target cache equals |P| and
   the draft cache does not exceed it. The retained draft prefix is valid because
   everything before correction matched its proposals.

The upstream-style cache control replays the whole verification batch on the
draft before trimming. It can waste draft work on rejected suffixes. The retained
path never promotes an unverified draft token into committed history.

Policies cannot mutate tokens or caches. Fixed caps disable confidence stopping.
Confidence stopping can produce fewer actual proposals. Acceptance compares the
accepted prefix against **actual** drafts. Scripted actions exist to test switching
and rollback, and are excluded from performance selection.

All requests clear KV, create a fresh upstream sampler/speculator and reset policy
state. Weights stay resident within a job process. Model load is measured once;
warm request time includes tokenization, reset, sampler/policy initialization,
prefill, all generation cycles, exploration, synchronization and fallback.
Generation starts immediately before evaluating the pending final prompt token
and ends after the last commit and cache repair. This boundary applies to all
paths; prompt prefill times exclude that final prompt token. A zero-token request
does setup/tokenization, no model work. EOG is retained in token-ID counts, but
is omitted from rendered text. First-token latency is the first completed
commit block (an upper bound on the first row's availability), not a streaming
timestamp. Text conversion and request-local teardown occur after generation
but within the driver's warm-request timer. `commit_boundary_ms` stops at the
last cache reconciliation. JSON serialization is outside warm timing;
whole-process wall time, including serialization, is retained by the Python runner.

Cycle logs are buffered, then serialized after generation. Minimal mode omits
cycle storage and serialization but still takes stage timestamps;
the logging ablation quantifies that narrower distinction. Do not describe it as
a fully instrumentation-free build.

Token IDs remain authoritative when a byte-fallback sequence ends midway
through a UTF-8 character. The CLI replaces invalid bytes only in display text
and records `text_utf8_repaired`; complete UTF-8 is preserved. This runs during
JSON output preparation, outside the warm-request timer and inside the runner's
whole-process time. It changes no sampling, acceptance, or cache state.

`getrusage(RUSAGE_SELF).ru_maxrss` supplies lifetime peak RSS (bytes on macOS,
KiB multiplied by 1024 on Linux). It includes loading and excluded warm-up,
is not per-request allocation or device VRAM, and cannot establish energy or
monetary savings. The practical target-only reference runs in its own process
without loading a draft. The resident control has both models loaded.

Synthetic GGUF fixtures are genuine tiny transformer forward passes with
controlled/random weights, manufactured only to exercise state transitions.
They are **not trained language models** or feasibility benchmarks. The suite
injects wrong proposal IDs to force zero/partial acceptance while retaining
upstream verification. Real-model differential tests remain an independent gate.
