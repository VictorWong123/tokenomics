# Compiler-IR locality diagnostic

The interleaving wrappers pass swapped grid coordinates to the original
mat-vec implementations. Source-level similarity does not establish what the
shader compiler emits. A separate five-minute standard macOS26 job generates
the exact owned shader input and requests textual LLVM-style IR with
`metal -S -emit-llvm -O3`, without running a GPU graph or loading a model.

The diagnostic compares four original/wrapper pairs after reversing the grid
coordinate permutation, renaming SSA/labels and the kernel symbol, resolving
attribute groups, and ignoring whitespace and metadata attachments. Arithmetic
operators, floating-point flags and address calculations remain in the text
comparison. Tests reject changed arithmetic, optimization flags, coordinates
and pointer offsets; missing functions are reported as incomplete.

This is an exploratory textual check, **not an equivalence certificate**.
Metadata and later function-constant/GPU specialization can affect execution.
Different IR can also implement equivalent arithmetic. The diagnostic cannot
replace direct M3 row comparisons or the complete trained exact-token gates.
The job reports differences/incomplete parsing explicitly rather than counting
them as GPU correctness. Raw IR and JSON diagnostics are retained as artifacts;
no weights, executables or environment dumps are uploaded.

The newer standard runner also runs a device-only capability probe to determine
whether its virtual Metal device advertises the SIMD reduction features missing
from the macOS15 runner. This does not start a benchmark, use Codespaces or
select a paid larger runner.
