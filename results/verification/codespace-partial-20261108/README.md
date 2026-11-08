# Bounded older development pilot — incomplete

[Run 42454095789](https://github.com/VictorWong123/tokenomics/actions/runs/42454095789)
at `87c7811ed5fe24024f50782a1307ecbcef52bdee` paused successfully after 11 of
108 comparison blocks. All 110 emitted-ID/prompt-ID/stop checks passed. The
hosted finalizer confirmed actual Codespace Shutdown.

Artifact ID 13609408452 contains the complete 636-member evidence archive.
The ZIP, inner archive and every manifest member were independently SHA-256
verified in Cloud. `summary.json` retains their identities. This directory
retains the original aggregate request/check records and frozen manifest;
`execution-manifest.json` identifies the full archived files. It deliberately
contains no `complete.json`: the full pilot did not complete.

Across the 11 observed prompt/cap cells, every drafting-policy throughput ratio
is below one; target-with-draft-resident is approximately neutral. The recorded
medians/min/max describe only available cells and repetitions. These are an
unbalanced prefix of a frozen random block order, not a full-suite aggregate,
uncertainty interval, development baseline selection or feasibility decision.
Do not treat missing comparisons as negative results.

The later driver fixes UTF-8 output boundaries and adds original-KV checkpoint
replay, and the warm-up protocol changed. Preserve this older point as
exploratory evidence. Never merge it with another binary/protocol, disable
fingerprint checks, or fabricate completion in order to analyze it as a pilot.

Reproduce the descriptive ratios with the repository virtual environment:

```python
from bench.common import read_jsonl
from analysis.report import cells
from statistics import median
from pathlib import Path

root = Path("results/verification/codespace-partial-20261108")
data = cells(read_jsonl(root / "requests.jsonl"))
for config in sorted({k[2] for k in data}):
    ratios = [v / data[(p, cap, "target")]
              for (p, cap, c), v in data.items() if c == config]
    print(config, len(ratios), min(ratios), median(ratios), max(ratios))
```
