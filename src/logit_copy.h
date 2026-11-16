#pragma once
#include <cstdint>
#include <span>
#include <vector>

namespace tokenomics {
struct LogitCopyStats {
    std::uint64_t source_candidates = 0, rank_checks = 0, logits_examined = 0;
    int selected_match_length = 0, hint_output_index = -1;
    bool hint_available = false;
};
// Prototype: chain-only, minimum-three known suffix, original top64 membership,
// no plain-copy fallback. Empty logits mean no past distribution is available.
// Accepted membership scans require finite logits and preserve first-index ties.
const char *logit_copy_backend();
void logit_copy_draft(std::span<const int> history, int pending, int cap,
                      std::span<const float> prior_logits, std::vector<int> &out,
                      LogitCopyStats &stats, int minimum = 3, int cutoff = 64);
} // namespace tokenomics
