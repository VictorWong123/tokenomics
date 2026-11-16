#include "logit_copy.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace tokenomics {
namespace {
bool top_member(std::span<const float> logits, int candidate, int cutoff, LogitCopyStats &stats) {
    ++stats.rank_checks;
    const float value = logits[candidate];
    if (!std::isfinite(value))
        return false;
    int better = 0;
    for (size_t id = 0; id < logits.size(); ++id) {
        ++stats.logits_examined;
        const float other = logits[id];
        if (!std::isfinite(other))
            return false;
        if (other > value || (other == value && id < size_t(candidate)))
            if (++better >= cutoff)
                return false;
    }
    return true;
}
} // namespace
void logit_copy_draft(std::span<const int> history, int pending, int cap,
                      std::span<const float> prior_logits, std::vector<int> &out,
                      LogitCopyStats &stats, int minimum, int cutoff) {
    if (cap < 0 || cap > 8 || minimum < 1 || minimum > 32 || cutoff < 1 || cutoff > 64)
        throw std::invalid_argument("invalid logit-copy settings");
    out.clear();
    stats = {};
    if (cap == 0 || prior_logits.empty())
        return;
    if (pending < 0 || size_t(pending) >= prior_logits.size())
        throw std::invalid_argument("logit-copy pending ID outside vocabulary");
    const size_t size = history.size() + 1;
    if (size <= size_t(minimum))
        return;
    const auto token = [&](size_t index) {
        return index < history.size() ? history[index] : pending;
    };
    struct Candidate {
        size_t boundary = 0;
        int id = -1;
    };
    std::array<Candidate, 33> best{};
    // Only the highest-ranked first token at each match length can win. If it
    // is outside top64, every other source at that length is outside too.
    for (size_t boundary = size - 1; boundary >= size_t(minimum); --boundary) {
        const int id = token(boundary);
        if (id < 0 || size_t(id) >= prior_logits.size())
            throw std::invalid_argument("logit-copy source ID outside vocabulary");
        if (id == pending || !std::isfinite(prior_logits[id]))
            continue;
        size_t length = 0;
        while (length < std::min(size_t(32), boundary) &&
               token(boundary - length - 1) == token(size - length - 1))
            ++length;
        if (length < size_t(minimum))
            continue;
        ++stats.source_candidates;
        auto &candidate = best[length];
        if (candidate.id < 0 || prior_logits[id] > prior_logits[candidate.id] ||
            (prior_logits[id] == prior_logits[candidate.id] && id < candidate.id))
            candidate = {boundary, id};
        // Descending boundary order retains the most recent source on equal IDs.
    }
    for (int length = 32; length >= minimum; --length) {
        const auto candidate = best[length];
        if (candidate.id < 0 || !top_member(prior_logits, candidate.id, cutoff, stats))
            continue;
        stats.selected_match_length = length;
        for (size_t i = candidate.boundary; i < std::min(size, candidate.boundary + size_t(cap));
             ++i)
            out.push_back(token(i));
        return;
    }
}
} // namespace tokenomics
