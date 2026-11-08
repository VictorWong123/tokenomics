# Historical trained-model correctness gate

[Run 42420516268](https://github.com/VictorWong123/tokenomics/actions/runs/42420516268),
source `d1bb6fb778695515614d21cf78959f63f544db7b`, runner 22, passed 272
candidate comparisons against ordinary target decoding. Its hosted finalizer
separately confirmed actual Codespace Shutdown. This is correctness evidence,
not a performance result or a validation of every later driver change.

The artifact ZIP and inner archive hashes match the original recorded digests.
All 32 manifest members have verified byte counts and SHA-256. This directory
retains those exact files plus the decoded workload report. In Cloud we also
independently checked all 272 candidate identities/execution settings, emitted
and prompt IDs, stop reasons, cycle commit totals and final cache positions.
See `independent-verification.json` for identities and limits.

The original `diagnostic/` is schema 1, with fresh-prefix, first-row probes and
an eight-token replay budget. It predates the corrected all-row/original-budget
diagnosis and the original-KV snapshot protocol. Preserve it as historical raw
evidence; it cannot clear current Pilot B. The separate `gate/` request
comparisons remain exact-token checks and were independently rechecked.

The models are Qwen2.5 0.5B draft and 1.5B target Q4_K_M. Their exact verified
hashes and public upstream revisions are in `models.lock.json` and
`gate/manifest.json`. Model weights, credentials and build trees are excluded.

`terminal-demo.txt` is a derived display of the scripted `development-qa-00`
request, repetition zero. Regenerate it with
`.venv/bin/python scripts/demo.py --recorded`. The command verifies original
file hashes, exact target IDs and cycle accounting before showing transitions;
it performs no inference. This is a scripted mechanics demo, not an adaptive
controller or a speedup result.

Recheck the emitted-token comparisons without downloading weights:

```python
from pathlib import Path
from bench.common import read_jsonl
from scripts.check_models import compare, check_cycles

gate = Path("results/verification/codespace-gate-20261108/gate")
refs = read_jsonl(gate / "reference.jsonl") + read_jsonl(gate / "boundary-reference.jsonl")
refs = {r["prompt_id"]: r for r in refs}
rows = read_jsonl(gate / "candidate.jsonl")
assert len(rows) == 272
assert all(compare(refs[r["prompt_id"]], r) is None for r in rows)
check_cycles(rows, read_jsonl(gate / "candidate.cycles.jsonl"))
```
