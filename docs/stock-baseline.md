# Exact stock executable: stopping mismatch reproduced

The unmodified pinned `llama-speculative-simple` can be built alongside the
adapter without enabling unrelated examples:

```bash
.venv/bin/cmake -S . -B build-cpu -DTOKENOMICS_BUILD_STOCK=ON
.venv/bin/cmake --build build-cpu --target llama-speculative-simple
```

This uses existing configured CPU settings and unchanged submodule sources.
The target is optional and disabled by default. The executable is
`build-cpu/stock-speculative-simple/llama-speculative-simple`.

A synthetic diagnostic on Codex Cloud reproduced a stopping-rule mismatch at
pin `fd27b2034905f8dadd8bd6d2cc44595f4929d391`. With draft cap 4, target/draft both
the generated `target.gguf`, greedy temperature-only sampling, CPU, one thread,
context 64, batch/ubatch 32, confidence stopping disabled and backend draft
sampling disabled:

| Requested output cap | Stock reported `n_predict` | Adapter output IDs |
| --- | ---: | ---: |
| 0 | 5 | 0 |
| 2 | 7 | 2 |
| 7 | 12 | 7 |

Exact commands, fixture/binary hashes and observations are retained in
`results/verification/pinned-stock/budget-observations.json`. Full local stdout/
stderr is under `results/local/cloud-continuation/stock-budget/`. These generated
fixtures are not trained models and the diagnostic makes no performance claim.

Source explanation: `examples/speculative-simple/speculative-simple.cpp` clamps
the remaining draft allowance to zero, but the draft interface does not treat
zero as a disable action. The loop also checks `n_predict > params.n_predict`
after committing a block, rather than stopping before an exhausted budget. The
adapter explicitly skips drafting when capacity is zero and bounds commits.
The stock counter increments the entire returned block before its EOG loop,
which is a further reason to validate actual output counts before using its
reported throughput. That EOG-count issue is a source-audit finding here, not an
additional claimed measurement.

Do not adjust `--n-predict` ad hoc, truncate stock text after timing, or label the
adapter's `cache=upstream` control as the exact stock executable. None restores
an identical measured stopping contract. The exact-stock matched performance
baseline remains unresolved. Any future patched diagnostic must be named as a
patch, preserve the original executable, and receive separate correctness and
timing validation. Stock text/log counts alone do not prove token-ID equivalence.
