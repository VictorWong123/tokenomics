# Conservative request-local suffix drafter

`--drafter suffix` is an opt-in, model-free proposal source. It requires a
matching suffix of at least five tokens, searches up to 32 tokens, and copies
only an already observed continuation. Longer matches win; equal-length
matches use the most recent occurrence. Overlap with the current suffix is
allowed, but no future or uncommitted token is read. There is no cross-request
cache, corpus, training step or second model.

The existing upstream `ngram` drafter is unchanged. The new drafter supplies
candidate IDs to the same upstream greedy verifier and existing commit/rollback
code. Fixed caps, scripted fallback and the cost-aware policy can use it. Its
state is derived entirely from the committed prefix, so request resets and
checkpoint replay do not require a separate retrieval-state snapshot.

This is a bounded backward scan, not an implementation of the full suffix
automaton or tree methods in [SAM Decoding](https://aclanthology.org/2025.acl-long.595/)
or [SuffixDecoding](https://proceedings.neurips.cc/paper_files/paper/2025/file/b7aea253ab34a773967f1e4cdea9e4fb-Paper-Conference.pdf).
Those methods motivate using match length as proposal evidence. Their reported
speedups do not transfer to this implementation by assumption.

## Why test this variant

Offline replay uses only the three predeclared development-fit prompts, 64
canonical tokens each, from the archived M3 0.5B and Linux 3B runs. It reproduces
all 30 measured fixed-policy aggregate work counts before comparing alternatives.
These archives used minimal logging, so there are no detailed cycle logs to
cross-check. Candidate retrieval sees the prompt and committed history only;
canonical future tokens are available only to the simulated verifier.

The fit exploration included upstream ngrams of length two and three, longest
suffix minima 2/3/4/5/6/8/12, and a two-occurrence consensus variant. Longer
matches without a minimum-confidence filter did not consistently help; consensus
almost eliminated useful proposals. Five-token minimum matching at cap one
removed all rejected proposals on both fit points:

| Fit operating point | Drafter, cap 1 | Cycles | Proposed | Accepted | Verified rows |
|---|---|---:|---:|---:|---:|
| M3 Metal, 0.5B | Upstream ngram | 159 | 37 | 33 | 196 |
| M3 Metal, 0.5B | Suffix, minimum 5 | 161 | 31 | 31 | 192 |
| Linux CPU, 3B | Upstream ngram | 159 | 38 | 33 | 197 |
| Linux CPU, 3B | Suffix, minimum 5 | 169 | 23 | 23 | 192 |

Each row emits the same 192 canonical tokens. The suffix variant saves less
cycle overhead but wastes fewer verification rows. This is **fit-set evidence,
not a measured speedup or an estimate of validation accuracy**. The threshold
was selected after examining these fit results. All alternatives and larger
caps remain in the replay output, including negative findings.

For intuition only, let a batched cycle cost `H + rows*R`, excluding drafting.
Compared with `N` ordinary cycles, a replay with `C` cycles and `V` verification
rows requires `(N-C)*H > (V-N)*R` to win. The output records the implied minimum
fixed-cost fraction. This optimistic model does not describe serial microbatches
or account for retrieval cost, graph changes or thermal effects; it is not a
runtime prediction.

Reproduce the complete offline comparison without loading any model:

```bash
.venv/bin/python -m analysis.proposal_replay \
  --out results/local/proposal-replay-01
```

## Local execution

Build the guarded Metal binary and run the correctness-first paired screen:

```bash
TOKENOMICS_BUILD_JOBS=2 bash scripts/bootstrap.sh metal
.venv/bin/python scripts/suffix_study.py \
  --model models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --out results/local/suffix-study-01
```

Use `--gate-only` to stop after correctness checks. The runner requires native
Apple Silicon, the pinned model, and the default guarded Metal marker. The
510-request gate exercises suffix proposals against ordinary decoding before
any timing. The 72-record screen uses the same executable for target-only,
ngram cap1, ngram cap8 and suffix cap1, with three development-fit prompts,
caps64/128 and three repetitions. Every block must emit identical IDs/stops.
It retains failed attempts, records executable/model/source identities and
before/after system metadata, and refuses changed binaries during the study.
The screen is exploratory because these prompts helped select the threshold.

For a single request after rebuilding:

```bash
build-metal/tokenomics \
  --target models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --gpu-layers 99 --threads 2 --context 2048 --batch 512 \
  --policy fixed --drafter suffix --k 1 --max-tokens 128 \
  --prompt 'Continue the sequence: abc abc abc' \
  --out results/local/suffix-request.jsonl \
  --cycles results/local/suffix-cycles.jsonl
```

Create the output directory first. This is a request example, not a performance
comparison. A paired local screen must keep the unchanged ordinary decoder,
ngram cap1 and ngram cap8 as baselines. Do not reuse ngram controller priors:
the frozen operating point includes the drafter, and suffix calibration must
be created separately. Action replay now reads that identity from the frozen
configurations rather than guessing from the absence of a draft model.

Current checks include 9,216 retrieval cases against an independent literal
reference, 1,134 runtime fixture cases, and a fixture-backed suffix calibration
and provenance check. These are CPU checks. The pinned trained-model download
was blocked by the Cloud network proxy (HTTP 403), so they do not establish
trained-model or Metal execution results. The existing Metal guard remains the
default unless its separate experimental option is explicitly selected.

## Calibration coverage

A fit configuration may declare `"replay_checkpoints": [9, 16, 32, 48]` before
measurements start. The schedule is frozen with the fit configuration and must
match during replay; it cannot be changed while resuming. Each checkpoint is
captured and replayed in its own process group, retaining only the same two
snapshots as before. The original default `[32, 96, 224, 400]` is preserved for
historical compatibility. Only checkpoints with room for a complete nine-token
continuation are eligible. A short fit run with no eligible checkpoints cannot
produce a valid calibration.

The historical cap64 fits provide only checkpoint32: three distinct states per
action, repeated three times. Only one M3 state and two Linux states produced
ngram proposals. Averages over these states cannot establish behavior across
all repetition patterns. The optional denser schedule improves observed-state
coverage; it does not guarantee a faster controller. It requires a new fit run
and newly frozen priors.

Audit any completed calibration without inference:

```bash
.venv/bin/python -m analysis.calibration_coverage \
  --fit PATH/fit --replay PATH/replay --drafter suffix \
  --out results/local/suffix-calibration-coverage.json
```

The audit rechecks immutable records, original checkpoint states and calibration
identity, and distinguishes repeated timing samples from distinct contexts.

The suffix correctness gate also writes a complete hashed receipt. Run
`python scripts/verify_metal_gate.py results/local/suffix-study-01/gate`
to check saved payload integrity and exact outputs without inference.
