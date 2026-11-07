# Pinned implementation audit

Pin: `fd27b2034905f8dadd8bd6d2cc44595f4929d391` (2026-11-07), an unmodified
Git submodule. CMake rejects a different revision. These findings come from
reading this revision, not from assuming the project-plan pseudocode is an API.

* `common/sampling.cpp`, `common_sampler_sample_and_accept_n`: samples each
  target logit row, returns the matching draft prefix plus one correction or
  bonus. Tokenomics calls this implementation in every path, including ordinary
  decoding with an empty draft. It does not introduce a second verifier.
* `common/speculative.cpp`, `common_speculative_impl_draft_simple`: greedy
  draft proposals use top-10 candidate normalization and the highest candidate.
  `p_min` compares against that **truncated** distribution, not a full-vocabulary
  calibrated probability. The confidence baseline is exactly this mechanism.
  `n_max` is a cap; zero is not a disable value, so Tokenomics explicitly skips
  drafting for target-only and zero remaining draft budget.
* `examples/speculative-simple/speculative-simple.cpp`: the example discards
  draft KV back to a checkpoint before verification, processes the complete
  target verification batch through the draft, then trims both caches. Our
  `cache=upstream` control reproduces this behavior for the supported
  full-attention independent-model mode; it is **not the stock executable**.
* Our `cache=retain` adapter keeps the valid draft prefix and lazily processes
  only missing committed tokens before the next proposal. It avoids computing
  rejected target verification suffixes on the draft. Both paths preserve the
  upstream verifier and proposals. Transition catch-up belongs to generation
  timing and is logged. This is an execution-path contribution, separate from
  the as-yet-unvalidated scheduling hypothesis.
* Upstream's compatibility check permits a small vocabulary-size difference and
  starts textual comparison after an initial token range. Tokenomics additionally
  requires identical vocabulary size, all token texts/attributes and EOG status.
  Each benchmark compares tokenized prompt IDs too. This deliberately rejects
  potentially remappable pairs instead of assuming compatibility.
* `llama_memory_seq_rm` returns a success flag. We check it and assert positions
  after every cycle. Only `COMMON_CONTEXT_SEQ_RM_TYPE_PART` is accepted. Recurrent,
  hybrid, sliding-window and checkpoint-only models are outside this adapter's
  contract. No context shifting is performed.
* `include/llama.h` documents automatic synchronization on reading logits.
  Verification timing ends after upstream sampling. Prefill/catch-up and drafting
  explicitly synchronize in **all** comparable adapter paths. Times are completed
  work, not just GPU submissions. Backend-specific performance still needs testing.
* `ngram-simple` uses prompt/history lookup without a second model. It is an
  upstream cheaper drafting mechanism available for the bounded negative-pilot
  investigation. Its lookup geometry is fixed at n=3, m=8, min_hits=1 here.

No submodule source changes are made. `src/runtime.cpp` owns request lifecycle,
cache reconciliation, decisions, limits, timing and structured results.

## Related work and claim boundary

Leviathan et al. (2023), Chen et al. (2023) establish speculative verification;
we restrict this project to greedy matching, not stochastic rejection sampling.
Mamou et al. (2024) studies confidence-based dynamic lookahead; our baseline
uses pinned llama.cpp's confidence stop and explicitly records its different
normalization. The PDF's two-success acceptance rule is a project-proposed
comparator, not claimed as a published algorithm. An online cost controller
would optimize committed tokens per completed wall time, including catch-up,
instead of acceptance alone. Neither scheduling nor draft-cache reuse is claimed
novel. The PDF's 2026 references have not been independently verified through
the restricted network; no novelty or conference-readiness claim relies on them.

Sources (replace the path after the pin to inspect any file above):
https://github.com/ggml-org/llama.cpp/tree/fd27b2034905f8dadd8bd6d2cc44595f4929d391
https://proceedings.mlr.press/v202/leviathan23a.html
https://arxiv.org/abs/2302.01318
https://proceedings.mlr.press/v262/mamou24a.html
