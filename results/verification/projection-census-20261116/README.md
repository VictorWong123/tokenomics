# Pinned projection census

Two standard public Ubuntu jobs read only a16MiB prefix of each locked GGUF,
parse its header/tensor directory and retain metadata. Full model SHA verification
is explicitly false here; no weights are saved. The prefix hashes match the
independent earlier inspections. Source `703f53e` produces all role/type/shape
variants, not a guess based on the `Q4_K_M` filename.

| Model | Logical encoded projection bytes | Output head share | FFN gate/up share | Current component shape coverage |
|---|---:|---:|---:|---:|
|0.5B|391573504|36.9%|36.7%|98.9%|
|1.5B|979517952|19.5%|44.3%|98.6%|

Both down-projection sets mix Q4_K and Q6_K. The0.5B gate/up matrices are
Q5_0[896,4864];1.5B uses Q4_K[1536,8960]. The expanded component grid includes
these FFN shapes and both down types, in addition to the attention and output
head classes. The selected shape/type classes account for over98% of each
model's logical projection payload; small K/V shapes remain outside this
standalone component grid and remain covered by the full backend/model gates.

These are encoded matrix payloads counted once per dense target forward.
Actual memory reads depend on cache reuse, repeated loads, quantization,
activation/KV traffic and backend dispatch. Tensor size fractions are not
runtime fractions, bandwidth measurements, speedups or an Amdahl bound.
Embedding lookup tensors and vector operations are excluded from projection
bytes and accounted for separately in each model's census.

Companion JSONs bind public run URLs, immutable model locks, prefix hashes,
source and all counted variants. `complete.json` binds these selected summaries
and the derived coverage calculation. The original full tensor-directory artifact
is not reproduced here; every role/quantization/shape count is printed in the
public run log and retained in these JSONs.
