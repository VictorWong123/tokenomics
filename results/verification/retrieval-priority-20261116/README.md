# Post-hoc suffix-priority combinations

This fixed follow-up keeps a minimum-five suffix proposal whenever available,
then uses either unchanged ngram3 or minimum-three/top64 guidance if it abstains.
The second rule reads a past rank row only when strong copying abstains and
capacity remains. No native drafter, fresh validation set or speed result is added.
These are already inspected development data, not the reserved90-prompt test set.

All five CPU collections retain16 cells each,384 per-prompt records and19616
computed cycles. Checkpoint `0452cb5` produces the retained candidate labels and
exactly the same numeric counts as the first `c99f971` run. All proposals/hints,
matched controls, model/platform/phase and protocol/data hashes are preserved.

| Dataset | Suffix-first guidance cap4 C/W | Suffix-only cap4 C/W | Ngram cap4 C/W |
|---|---:|---:|---:|
|0.5B x86 fit,192 tokens|140/4|142/1|140/18|
|1.5B x86 fit,192 tokens|140/20|153/13|141/40|
|0.5B x86 validation,384 tokens|285/14|289/4|279/22|
|1.5B x86 validation,384 tokens|324/32|342/30|315/36|
|0.5B ARM validation,384 tokens|277/14|282/9|269/20|

C means target cycles; W means rows beyond committed output. Giving suffix
priority can recover cycles lost by the original guidance filter, but trades
waste against stronger controls and adds unmeasured guidance cost. At cap1 the
combination improves work over suffix-only on validation; that remains a
post-hoc work point, not a speed/generalization result. Suffix-first ngram is
identical to ngram in nearly every cell, with a small fit-only1.5B/cap4 change.
Neither combination establishes a general baseline win or merits another native
mode before actual M3 verification cost is known.

Reproduce with `analysis.transition_replay --family priority --collection PATH
--out FRESH`, using one of the sealed rank collections. Default `--family
transition` retains the earlier frozen linear-transition comparison.
