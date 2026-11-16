# Exploratory linear transition results

The fixed14-cell exploratory comparison runs on five already observed CPU
collections. These are336 per-prompt records and17416 computed cycles; all
proposals and counts are retained. Checkpoint `a5318a9` defines the executed
code/protocol. This is post-hoc development exploration, not a new validation,
held-out test or model-speed measurement.

| Dataset | Two-token transition cap8 C/W | Ngram cap8 C/W | Previous-second-logit cap1 C/W |
|---|---:|---:|---:|
|0.5B x86 fit,192 tokens|130/93|133/25|184/170|
|1.5B x86 fit,192 tokens|125/99|134/70|181/164|
|0.5B x86 validation,384 tokens|266/234|267/38|366/336|
|1.5B x86 validation,384 tokens|301/257|311/93|364/332|
|0.5B ARM validation,384 tokens|256/230|257/42|365/335|

C is target cycles and W is extra verification rows beyond committed output.
The two-token graph sometimes reduces cycles, but wastes many more rows. The
one-token graph's complete grid is worse still. The prior distribution's second
candidate is a poor direct next-output guess: its drafted-token acceptance is
about5% on the development-validation collections. These results do not support
adding either native drafter. No ratio of work counts is presented as speed.

This evaluates one linear walk through observed transitions. It does not evaluate
Token Recycling's branching candidates, rejected-path cache updates, hot-start
vocabulary table or tree attention. It also does not reproduce LogitSpec.
Nothing here disproves those published methods on their own hardware/models.

Reproduce one collection with `analysis.transition_replay`; see
[design and commands](../../../docs/transition-replay.md). The sealed report
records model/platform/phase and the original collection hash. CPU trajectories
remain separate, and the reserved90-prompt test set is untouched.
