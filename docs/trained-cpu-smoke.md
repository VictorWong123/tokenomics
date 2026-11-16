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

The standard public Ubuntu workflow builds a default guarded executable,
copies it, and rebuilds the sampler-only candidate in the same build tree.
This reuses compiled dependencies without overwriting the retained reference.
It first tests the orchestration on synthetic fixtures, then downloads the
locked public model and runs the trained check. It stops after30 minutes and
uploads only explicit correctness/configuration files for seven days. Model
weights and executables are excluded. It uses GitHub Actions, not Codespaces.

The local orchestration test passes160 fixture comparisons and confirms that
passing a guarded executable as the candidate fails its sampler marker check.
These fixture results do not establish the trained model result; the hosted
run must finish separately.
