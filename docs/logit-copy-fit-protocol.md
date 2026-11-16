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

## Completed CPU fit

Both pinned models produce192 canonical output tokens across the three fit
prompts. Each collection verifies observer/ordinary IDs and stops, full model
SHA, and192 stable top64 rows. Imported numeric data match the sealed workflow
hashes. Each model's36 guided cells plus12 controls are retained with144
per-prompt records; there are14041 computed cycle/proposal records across both.
These are independently recorded Linux CPU operating points, not M3 data.

| CPU model / strategy | Cap | Cycles | Wasted verification rows |
|---|---:|---:|---:|
|0.5B upstream ngram3 |8|133|25|
|0.5B raw minimum-three suffix |8|133|27|
|0.5B guided minimum-three, top64, no fallback |8|133|3|
|0.5B raw minimum-five suffix |1|161|0|
|0.5B guided minimum-three, top64, no fallback |4|144|0|
|1.5B upstream ngram3 |4|141|40|
|1.5B guided minimum-three, top64, no fallback |4|141|13|
|1.5B guided minimum-one, top64, no fallback |4|128|28|

The conservative guided points reduce wasted rows at equal cycle counts; more
aggressive points can change both. Computing rank membership is extra work that
plain copying does not incur. Query-size histograms remain necessary: fewer
rows or cycles do not establish lower elapsed time. The full grid/frontier,
including aliases, is retained, with no runtime deployment selected.

The new0.5B CPU canonical IDs match the archived M3 fit on code/summary but
differ on QA despite matching prompt IDs. Its top64 distributions have not been
compared with Metal. No cross-hardware rank equivalence or speed transfer is
claimed. [Evidence and raw numeric data](../results/verification/logit-copy-fit-20261116).

A separate post-fit development-validation protocol now freezes the conservative
minimum-three/top64/no-fallback candidate at caps1/4/8 on six other development
prompts. This selects a candidate for further testing, not deployment. Before
inference integration, a native prototype must reproduce the Python proposals
and measure host membership cost. The proposed implementation tests whether the
best source token at each match length is globally top64, instead of sorting
all151936 vocabulary entries. This preserves ranking semantics while allowing
quick rejection after64 better entries. No extra model forward is required.

The first unselected native prototype passes59022 independent sorted-reference
cases in Release and ASAN/UBSAN and reproduces108 per-prompt Python work records
across both models. These checks embed recorded ranks in synthetic logits; they
do not run models. Its192 raw component pairs expose a slow rejection case:
a just-outside-top64 source token is rescanned nine times. At vocabulary151936,
that stress case takes roughly1.4ms on this Linux CPU. The prototype is not yet
called by inference. The next optimization caches repeated membership checks;
this negative cost result is retained with its original source/binary hashes.

The collector/importer now distinguish the frozen fit and validation protocols
by explicit phase and separate SHA pins. Validation collects six prompts and
replays exactly three guided cap settings, six fixed retrieval controls, and
ordinary decoding; it does not rerun the exploratory parameter grid. A synthetic
phase test checks384 rank rows,60 per-prompt records,10 totals and ordinary
zero-waste accounting. Old fit receipts remain readable and preserve their grid.

## Frozen development validation

The six separate development prompts produce384 tokens per model. Only the
three predeclared guided caps and seven controls were replayed; no threshold
retuning or held-out evaluation followed. On0.5B, guided cap4 C297/W10 loses
to suffix cap4 C289/W4, and guided cap8 C283/W27 loses to suffix cap8 C279/W15.
On1.5B, guided cap4 C326/W26 improves on suffix cap4 C342/W30 but trades11
extra cycles for10 fewer wasted rows versus ngram cap4 C315/W36. These exclude
ranking cost and do not establish lower latency. The runtime prototype stays
default-OFF. Full10-cell results and all6447 validation cycle records are
retained separately from fit. No new guided parameter search used validation.
