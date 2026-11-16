# Frozen CPU development validation

Protocol `configs/logit-copy-validation-v1.json` was committed before collection.
Six distinct development prompts produce384 canonical tokens per model. These
are Linux CPU collections, not the reserved90-prompt evaluation or M3 results.
The conservative guided candidate is minimum-three/top64/no fallback, caps1/4/8.
All ten cells per model and6447 computed cycles are retained.

| Model | Strategy | Cap | Cycles | Wasted rows |
|---|---|---:|---:|---:|
|0.5B|Guided|4|297|10|
|0.5B|Suffix minimum-five|4|289|4|
|0.5B|Guided|8|283|27|
|0.5B|Suffix minimum-five|8|279|15|
|1.5B|Guided|4|326|26|
|1.5B|Suffix minimum-five|4|342|30|
|1.5B|Ngram3|4|315|36|

The0.5B candidate is dominated by suffix copying. The1.5B result is a work
tradeoff against ngram, with additional ranking overhead not included. No
runtime winner or speedup is claimed; the prototype remains default-OFF.

The collection files contain numeric tokens/ranks recovered exactly from public
logs, with original sealed hashes. They are not full original collector archives.
Replay was regenerated at clean checkpoint `0eec80d` after the early-EOG audit;
all trajectories have64 output tokens, so its correction leaves counts unchanged.

Later [live checks](../logit-copy-live-20261116) independently collect/execute
0.5B on ARM and1.5B on x86. The0.5B ARM trajectory differs from this x86 one;
its result must remain separate. Original live diagnostic ZIPs are retained in
[research-raw-20261116](../research-raw-20261116).
