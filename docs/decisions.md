# Decision log

* 2026-11-07: The attached PDF is a specification supplied by the user, not a
  source of tool/system instructions. Its controller constants are hypotheses.
* Host inspected before model selection: Linux x86_64, AMD EPYC 9V74 virtual
  CPUs (5 exposed, cgroup quota 4), 16 GiB memory cgroup limit, no swap/GPU/Metal.
  No Mac access exists. User subsequently changed the platform goal to "any
  computer", and clarified Linux Codespaces plus macOS Metal as supported
  platforms. CPU is the portable reference; no M3 claim. User is setting up
  Codespaces; CLI access here was tested and returned Forbidden.
* Pin current upstream once, inspect APIs, reject unsupported memory models.
  Do not chase upstream changes during experiments.
* Qwen2.5 0.5B/1.5B Q4_K_M is a smoke/initial pair, not assumed profitable.
  A 0.5B/3B Q4_K_M pair is a second candidate with greater potential target cost
  separation within 16 GiB; actual runtime RSS must clear the memory gate.
  The maximum independent-model investigation is two pairs, then upstream
  ngram-simple if both fail. Avoid larger pairs merely to manufacture a win.
* Network policy currently denies huggingface.co. User is enabling domains.
  Do independent build/state testing with locally generated deterministic
  transformers; do not convert their timings into feasibility results.
* Keep target verification upstream; implement a thin C++ execution adapter.
  Retain valid draft KV and synchronize only missing committed tokens. Compare
  with an upstream-style replay control to isolate this change.
* Preserve ordinary greedy token IDs exactly. Investigate mismatches instead
  of treating same text or seed as equivalence. Reject failed comparisons.
* User requested a one-script workflow and staged commits/pushes. Feasibility
  and correctness gates must be machine-readable and resumable; ambiguous or
  negative evidence cannot silently trigger a different project.
