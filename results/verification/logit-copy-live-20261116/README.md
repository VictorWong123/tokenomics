# Learned live guided decoding checks

Both public workflows at source `f9ecb46` pass60 actual requests: six frozen
development-validation prompts times ordinary, ngram1/4/8, suffix1/4/8 and
guided1/4/8. Every prompt/token ID, stop, cycle/drafted/accepted/verified count
matches the canonical prediction. Detailed hints reference only the last
committed predicting row. Current corrected replay independently reproduces
all totals from the imported numeric datasets.

| Model/platform | Guided cap4 C/W | Suffix cap4 C/W | Ngram cap4 C/W |
|---|---:|---:|---:|
|0.5B Apple ARM CPU/NEON|290/11|282/9|269/20|
|1.5B Linux x86 CPU/SSE2|326/26|342/30|315/36|

Each model emits384 tokens per condition. C means target cycles; W means
verification rows beyond committed tokens. The0.5B ARM canonical trajectory
and work differ from the earlier x86 collection, while the suffix dominance
conclusion persists. No CPU trajectory is substituted for M3 Metal.

Original complete diagnostic archives and an offline audit are preserved in
[research-raw-20261116](../research-raw-20261116). This directory holds selected
public summaries and exact numeric data with corrected replay. Imported local
collection provenance describes the scope of log recovery; use the original
ZIPs for full collector/live records.

This establishes implementation agreement for these CPU development inputs,
not speed or generalization. Ranking overhead is additional, all timing values
are diagnostic, and the prototype remains disabled by default.
