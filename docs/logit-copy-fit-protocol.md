# Causal logit-guided copying: exploratory fit protocol

[Token Recycling, ACL2025](https://aclanthology.org/2025.acl-long.338.pdf) stores
previous target candidates in a token adjacency structure and verifies draft
trees. It can propose combinations absent from the observed text, unlike plain
prompt lookup. Our current verifier supports one linear sequence; reproducing
its branching method would require new masking and KV management. Its published
speedups therefore do not transfer directly to this backend.

[LogitSpec](https://arxiv.org/html/2507.01449v1) uses a previous target logit's
high-ranked candidates to suggest the token after its selected token, retrieves
continuations, and verifies multiple branches. Its main experiments use FP16
CUDA models. The useful transferable question is whether past target rankings
help choose among request-local copy continuations. A linear approximation is
not a reproduction of the paper's tree method.

`configs/logit-copy-fit-v1.json` is frozen before collecting new model logits.
It uses only the same three development-fit prompts,64 output tokens each,
with no reserved held-out inputs. An opt-in diagnostic collector will record
ordinary target IDs and stable top64 candidates, using the guarded CPU backend
and original sampler. Source, model, collector and protocol hashes accompany
the data. Diagnostic elapsed time is not a benchmark.

At each replay cycle, only the predicting row for the already committed pending
token may guide drafting; the first cycle has no hint. Exclude that pending
token itself from the hints. Search prior observed context for continuations
whose first token is hinted, choosing the longest matching known suffix, then
lowest hint rank, then most recent occurrence. Copy only already observed tokens.
The frozen grid varies minimum suffix1/3/5, top8/64 hint cutoffs, optional plain
copy fallback, and caps1/4/8:36 cells per model. Matched raw suffixes and upstream
ngram3 are controls. All cells and traces remain visible; none is selected for
deployment. Independent exhaustive small-sequence tests must check matching and
causal hint availability before replay.

This collects canonical-prefix logits only, not logits from rejected speculative
branches. Replay reports cycles, accepted/drafted/wasted rows and query shapes,
with no fitted latency or generalization claim. Any useful point still needs
runtime integration, actual speculative checks and an M3 comparison against the
optimized ordinary decoder.

The collector is compiled only with `TOKENOMICS_LOGIT_COLLECTOR=ON`, with the
original guarded CPU/sampler profile. Its read-only observer is absent from
normal benchmark builds. Each prompt runs once without and once with the
observer; IDs and stops must agree, and each stable top64 row must select the
same token as the original sampler. The synthetic fixture checks first-index
ties explicitly. Collection refuses model/protocol hash changes and fresh-output
overwrite. A bounded standard public Ubuntu workflow is capped at25 minutes.
The log transport includes only compressed numeric data from these fixed public
prompts and selected hashes, never weights or environment variables.

The replay implementation now passes118044 small guided cases against independent
prefix enumeration, plus matched raw-suffix checks and explicit past-row access
guards. A synthetic collection passes compressed log import, exact raw data-hash
reconstruction, the full48-cell guided/control replay, and incomplete/mutated
payload rejection. All125 Python tests pass with the collector fixture enabled
(eight unrelated optional skips). This implementation checkpoint precedes reading
the learned fit results.

The importer preserves numeric rank data exactly and verifies its sealed dataset
hash against the public workflow receipt. Its local import receipt explicitly
states that original stdout/jobs/manifest files were not independently downloaded;
their original hashes remain available. Numeric data, provenance and a new local
seal are retained separately. This is more limited than a full artifact audit.
