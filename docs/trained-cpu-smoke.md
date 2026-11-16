# Bounded trained-model CPU smoke

`scripts/trained_cpu_smoke.py` compares ordinary guarded decoding, model-free
ngram/suffix proposals, scripted fallback, and the optional greedy-only sampler
using the pinned Qwen2.5 0.5B weights. It verifies full model SHA256 before
running. This is a **correctness smoke**, with no performance or M3 claim.

Four development prompts use caps128/64/64/64. Four tiny output budgets and two
255/511-token prefix boundaries add10 cases in total. Each sampler profile runs
eight configurations (ordinary, ngram caps1/4/8, suffix caps1/4/8, and scripted
ngram fallback). All160 resulting comparisons must match the ordinary
reference's exact prompt IDs, output IDs and stop reason. Both drafter families
must actually propose and accept tokens. Paired sampler profiles must perform
the same logical work. Cycle logs are audited, source/binary identities remain
fixed, and the first failing record is retained. Reserved held-out prompts are
not read.

The standard public Ubuntu or Apple ARM macOS workflow builds a default guarded executable,
copies it, and rebuilds the sampler-only candidate in the same build tree.
This reuses compiled dependencies without overwriting the retained reference.
It first tests the orchestration on synthetic fixtures, then downloads the
locked public model and runs the trained check. It stops after30 minutes and
uploads only explicit correctness/configuration files for seven days. Model
weights and executables are excluded. It uses GitHub Actions, not Codespaces.
The `trained_runner` dispatch input selects `ubuntu-24.04` or `macos-15`; both
are standard public runners. The Apple runner checks ARM CPU behavior, not
Metal GPU behavior or M3 performance. A separate sampler component benchmark
retains all48 alternating paired timings, including candidate-array preparation;
it is not an end-to-end generation benchmark.

The local orchestration test passes160 fixture comparisons and confirms that
passing a guarded executable as the candidate fails its sampler marker check.
These fixture results are separate from the learned-model result.

The [hosted learned-model run](https://github.com/VictorWong123/tokenomics/actions/runs/43811081794)
also passes160 comparisons, including nonzero accepted ngram and suffix
proposals and identical work between sampler profiles. Across its348 canonical
output tokens, suffix cap4 uses296 cycles/353 verification rows; ngram cap8 uses
284 cycles/422 rows. These aggregate smoke counts include tiny budgets and
artificial prefix-boundary cases. They are not a workload performance estimate.
Only the printed aggregate summary is retained locally; full raw payloads
remain in the workflow artifact. No Metal graph or M3 speed claim is made.

The [standard Apple ARM run](https://github.com/VictorWong123/tokenomics/actions/runs/43826134066)
also passes160 learned comparisons and equal work between sampler profiles.
Its348 canonical output tokens use261 cycles/432 verification rows with ngram8
and283 cycles/364 rows with suffix4. These differ from the x86 aggregate counts;
each run checks against its own same-machine ordinary reference. Cross-ISA
token agreement was not tested. The native sampler equivalence executable and
synthetic orchestration tests also pass. This remains CPU correctness evidence.
