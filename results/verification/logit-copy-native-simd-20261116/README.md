# Native guided-proposal component

Retains384 raw component pairs (192 SSE2 and192 scalar),108 complete replay
records and5546 per-cycle proposals/hint indices checked against Python.
`summary.json` binds compiled source and actual executable hashes from checkpoint
`f9ecb46`, before the later replay-only early-EOG harness restriction.

At vocabulary151936 and history131, median complete proposal construction is
43.63µs for a hit and45.10µs for a just-outside-top64 rejection on this Linux
x86 host. At history2051, those values are56.24µs/54.95µs. Each repeated source
candidate token is ranked once. The earlier nine-scan scalar failure is retained
separately in `logit-copy-native-scalar-20261116`.

These are cache-warm synthetic host measurements, including source search and
allocation. The sorted reference also enumerates prefixes; ratios are not
membership-only speedups. None establishes generation throughput, ARM/NEON
cost or M3/Metal performance. See the default-OFF runtime documentation and the
negative/mixed development validation before considering deployment.
