# Frequency-filtered copying: exploratory fit protocol

The existing copy oracle shows opportunities below the conservative five-token
minimum match. This experiment asks whether observed continuation frequency
can filter weak matches without learning from future tokens. It is a chain-only,
request-local hypothesis; it is not a reproduction of the published suffix-tree
systems or their cross-request/tree verification methods.

The grid is frozen in `configs/frequency-copy-fit-v1.json` before execution:
four minimum match lengths, two support counts, three joint confidence ratios
and five caps. Search tries longest suffixes first. A next token must pass both
the support count and the ratio of that entire continuation prefix's occurrences
to the original matched-suffix count. Later missing/censored continuation tokens
cannot improve the ratio. Frequency ties prefer the most recent occurrence.
Only an empty first step permits backing off to a shorter suffix.

Replay uses the same three archived development-fit trajectories per operating
point,64 tokens each, with ordinary/ngram/suffix controls re-audited first.
Proposals read only the current prompt, committed output and pending token.
No single grid cell is selected for deployment. The report retains all cells,
unique cycle/wasted-row Pareto points and aliases; grid cells sharing the same
trajectory are not independent experiments. There is no model inference,
latency estimate, validation result or held-out claim.

The trace retains actual proposed token IDs, effective budgets and accepted
counts. Histograms of verification query-row counts remain available alongside
aggregate work: equal cycle/waste counts can still have different kernel/KV
costs. These histograms are descriptive, without a fitted latency model.
