# Original-state checkpoint checks (Cloud)

This is synthetic transformer evidence, not trained-model performance.
The C++ engine uses full sequence-state host copies, checks serialized sizes
and restored positions, shares one prepared draft state across actions, and
bounds state payload at 1 GiB/eight checkpoints. Ordinary requests release it.

- 378 differential fixture cases; saved-state tests compare exact tokens and
  first-row logits across all fixed caps, including tiled-prefill history.
- 25 minimal/checkpoint and 18 UTF-8 boundary cases also pass.
- Four Release CTest suites and four address/undefined/leak sanitizer suites
  pass. The extended-prefix runtime sanitizer check also passes.
- Full fixture workflow: 24 blocks/168 exact-ID requests. Original-state replay
  checks 264 canonical action outputs in 12 groups, with 12 trace captures.
- Standalone `python -m bench.replay` executes the same schema-3 path.

Reproduce with the standard bootstrap and fixture command in README. This
verification used `results/local/checkpoint-v3-final` and the standalone replay
at `results/local/checkpoint-v3-cli`. `verification.json` records the exact
owned-source and binary fingerprints. Capture/restore times are nested inside
diagnostic prompt/generation timings; they are not a controller speedup.
