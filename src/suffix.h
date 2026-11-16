#pragma once
#include <span>
#include <vector>

namespace tokenomics {
// Request-local suffix retrieval. Frozen minimum match 5, maximum match 32;
// ties favor the most recent earlier occurrence. Only observed tokens are read.
void suffix_draft(std::span<const int> history, int pending, int cap, std::vector<int> &out);
} // namespace tokenomics
