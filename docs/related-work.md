# Related work and the implemented baselines

Primary sources inspected on 8 November 2026. These describe other systems;
their reported gains are not Tokenomics measurements. This comparison is a
foundation for the feasibility investigation, not an exhaustive novelty review.

| Source | Available signals and actions | Cost assumptions and evaluation | Consequence here |
|---|---|---|---|
| [Leviathan, Kalman and Matias, ICML 2023](https://proceedings.mlr.press/v202/leviathan23a/leviathan23a.pdf), especially §§3.3–3.5 | Approximate/target distributions; a chosen lookahead γ and speculative verification | The analytic speedup depends on acceptance α and draft/target cost ratio c, assumes sufficient parallel compute and long enough generation. Evaluation uses T5-XXL. | High acceptance alone does not establish acceleration. Measure verification cost across widths on the actual backend. Our greedy adapter reuses upstream verification; it does not implement their general stochastic algorithm. |
| [Chen et al., 2023](https://arxiv.org/pdf/2302.01318), especially §§3–4 | Draft continuations and target scores; modified rejection sampling | Chinchilla evaluation uses distributed TPU v4, XSum and HumanEval at batch one. The authors explicitly discuss numerical differences between ordinary and speculative graphs, including greedy divergence. | Distribution preservation within numerics is different from our declared exact-ID contract. Our CPU kernel profile must disclose its cost; published TPU speedups do not predict this host. |
| [Mamou et al., DISCO, ENLSP 2024](https://neurips2024-enlsp.github.io/papers/paper_19.pdf), especially §§2–4 | Top draft probabilities, entropy and token position feed a trained classifier that decides continue/verify | Compares dynamic decisions with tuned static lookahead and heuristics across code, summarization and instruction tasks. Training and classifier execution are part of the method. | Our pinned confidence stop is a comparator, not a DISCO implementation. Request-level variation or hindsight best actions do not prove deployable cycle adaptation. |
| [Intel/Hugging Face implementation article, October 2024](https://huggingface.co/blog/dynamic_speculation_lookahead) | Confidence threshold can stop drafting before the configured cap; contrasts this with acceptance-length heuristics and DISCO | Behavior belongs to the documented Transformers implementation/version. | Record the precise pinned llama.cpp stopping/normalization rules instead of implying the libraries implement identical policies. |
| [Mainardi, Sandikci and Vanschoren, EACL 2026](https://aclanthology.org/2026.eacl-long.255.pdf), especially §§3–4 and limitations | Compares independent, self and retrieval drafting; measures component costs and generated length per iteration | Includes the same Qwen2.5 0.5B→1.5B family pair. Uses float16, greedy batch-one generation, 480 SpecBench samples, three runs and a 24 GB NVIDIA A5500. Finds costly independent drafting and stronger cheap retrieval methods. | This motivates the ngram screen, but neither its float16 GPU results nor its retrieval implementation establish our Q4 CPU result. No additional training is authorized by citing this work. |
| [Zhang et al., Learning to Draft, arXiv v1, March 2026](https://arxiv.org/html/2603.01639v1), especially §§3–4 and appendices A/C | Learned policies use draft probabilities, depth and context length to choose tree depth and verification size | Optimizes cycle throughput with reinforcement learning; experiments use NVIDIA A100 and evaluate policy overhead. | Optimizing throughput rather than acceptance is already prior work. Our proposed untrained per-action estimates, chain caps and fallback are a narrower systems experiment, not a new general objective. The arXiv record was verified; venue acceptance is not established here. |
| [Yemets et al., MacPaw Research, September 2026](https://research.macpaw.com/publications/llm-optimizations-apple-silicon), speculative-decoding section | Evaluates classic, ngram and tree methods as full deployed configurations | Its matched 4-bit Qwen3-4B comparison uses 100 conversational prompts, greedy single-request inference on M2 Pro/M3 Ultra; evaluated speculative methods lose in that workload. | Quantization, runtime, device and workload all matter. This is the authors' published result, not an M3 Air test or a prediction of ours. |

The exact implementation comparison comes from the checked-out
[llama.cpp pin](https://github.com/ggml-org/llama.cpp/tree/fd27b2034905f8dadd8bd6d2cc44595f4929d391),
not moving documentation. See [upstream audit](upstream.md) for the greedy
sampler, truncated top-10 confidence normalization, stock stopping mismatch,
cache reconciliation and ngram geometry. In particular, the model confidence
baseline and upstream variable-length ngram comparator must both be included
when judging the corresponding scheduling hypothesis.

The current contribution is the owned C++ execution/state adapter, experimental
cost-aware controller, differential checks and reproducible measured protocol.
The [bounded controller study](controller-results.md) is complete; it beats the
cap8 ngram comparator but does not accelerate ordinary decoding. Cache reuse and
scheduling are not claimed novel. Held-out gains and cross-platform performance
remain unestablished.
