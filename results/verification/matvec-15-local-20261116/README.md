# Model-aware matrix diagnostic checks

The current component has five shape/type classes per model. Its expanded
native source is checkpoint `5ce28cf`; the earlier `c7e4232` receipt covered the
initial three-shape 1.5B version. Both versions' observations remain distinct.

`expanded/` retains actual Linux Release and ASAN/UBSAN outputs, native geometry
listings, executable/source hashes and selected build configurations. Each mode
passes 20 small 0.5B arithmetic gates/10240 compared values, and ten 1.5B gates/5120
values. These 256×128 CPU matrices check allocation, views, quantization and row
comparisons; they are not the large target matrices or timing measurements.
The native-only contract additionally builds/runs all five pure C++ suites in
a fresh configuration with Python package finding disabled.

[Native Apple run 43870311106](https://github.com/VictorWong123/tokenomics/actions/runs/43870311106)
passes 13 stages at `5ce28cf`, including C++/Metal compilation, the 1072-case CPU
backend-row suite, both small component modes and 1.5B geometry output.
`native-expanded-preflight.json` retains the public source/run, stage exit codes,
log hashes and inference binary identity. That binary is byte-identical to the
initial 1.5B diagnostic run `43867839119`; changing only component geometry does
not change inference. These selected receipts do not reproduce all stage logs.

The hosted virtual GPU lacks SIMDgroup reduction support. No large Metal matrix,
full trained Metal gate or M3 timing was executed. The real Mac must run the
[component protocol](../../../docs/metal-matvec-component.md) and subsequent
trained gates. The current default study requires 6400 timing pairs/400 gates;
`--model-id qwen25-15` requires 3200 pairs/200 gates, each covering 25 query/shape
cells. Those are required future study sizes, not observed Metal measurements.
