# Experimental past-logit copy drafter

The frozen development validation does not support replacing the existing
retrieval drafters generally. In0.5B, guided caps 4/8 are dominated by the
minimum-five suffix controls in cycles and wasted rows. In1.5B, guided cap4
improves on suffix copying but trades more cycles for fewer wasted rows than
ngram. Ranking cost is additional. This prototype therefore defaults OFF as
`TOKENOMICS_LOGIT_COPY`; standard Metal studies explicitly disable it.

When enabled, `--drafter logit-copy` proposes a single request-local chain using
minimum-three/max32 known suffixes and the prior distribution's top64 members.
There is no plain-copy fallback. The first cycle has no hint. Later cycles use
only the row that predicted the last committed correction/bonus token; rejected
suffix rows are never used as that hint. The distribution is consumed before
the next target decode overwrites the output buffers. Each request resets this
row index, including diagnostic replay requests. No draft model, extra target
forward, cross-request data or full-vocabulary sort is needed.

At each known match length, only the highest-ranked first source token can win.
If that token is outside top64, the other candidates at that length are also
outside. Rank checks preserve first-index ties, cache repeated token IDs within
a call, and use integer-counted SSE2/NEON comparisons where available. A scalar
fallback is retained. Accepted membership checks require finite logits. Model
suppression is rejected for this prototype. Upstream greedy sample/accept,
correction, bonus, stopping and KV rollback remain responsible for correctness.

Detailed cycle logs expose hint availability/output index, matching-source
counts, membership checks, examined logits and selected match length. Request
totals expose the frozen algorithm settings and summed work. They allow the
live checker to reject future/stale hint use and telemetry mismatches.

The implementation passes59082 independent native sorted-reference cases,
including ties, rank boundaries, signed zeros and subnormals. Native replay
matches5546 individual proposal/hint records across both fit collections.
Enabled runtime fixtures cover1362 cases, including injected zero/partial/full
acceptance, action0 re-entry, request reset, EOS and context limits. Release and
ASAN/UBSAN suites pass. These local native checks use Linux x86_64; ARM/NEON execution is checked
separately on the hosted runner. Two live synthetic suites cover full-budget
and early-EOG trajectories. Learned CPU live validation also passes 60 requests
per model: 0.5B ARM/NEON and 1.5B x86/SSE2, with exact IDs/stops/work and causal
hint ages. The 0.5B ARM trajectory differs from x86, but suffix still dominates
guided caps 4/8. Original diagnostic ZIPs and an independent offline audit are
[retained](../results/verification/research-raw-20261116/README.md).
These results do not establish trained Metal or M3 speed.

The host component benchmark compares complete proposal construction with a
sorted-vocabulary/prefix-enumeration reference. Its synthetic cache-warm results
include source search and allocations; they are not membership-only timings or
model throughput. The scalar repeated-scan failure is retained before its fix.

For CPU research, build a separate directory with
`-DTOKENOMICS_LOGIT_COPY=ON`, original guarded CPU enabled, and the sampler/
stage-timing experiments OFF. The ordinary executable then accepts
`--drafter logit-copy --policy fixed --k 4`. `scripts/check_logit_copy.py` compares
live decoding against a sealed, predeclared validation rank collection; it
checks all 60 ordinary/ngram/suffix/guided requests and audits hint ages. The
`logit-live` workflow collects and checks the same frozen validation on a bounded
standard public CPU runner. Reserved held-out prompts remain untouched.
