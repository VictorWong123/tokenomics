#include "logit_copy.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#if !defined(TOKENOMICS_LOGIT_COPY_SCALAR) && defined(__SSE2__)
#include <emmintrin.h>
#elif !defined(TOKENOMICS_LOGIT_COPY_SCALAR) && defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace tokenomics {
namespace {
bool top_member(std::span<const float> logits, int candidate, int cutoff, LogitCopyStats &stats) {
    ++stats.rank_checks;
    const float value = logits[candidate];
    if (!std::isfinite(value))
        return false;
    const int count = int(logits.size());
    int better = 0, id = 0;
#if !defined(TOKENOMICS_LOGIT_COPY_SCALAR) && defined(__SSE2__)
    const auto value4 = _mm_set1_ps(value);
    const auto candidate4 = _mm_set1_epi32(candidate);
    const auto exponent = _mm_set1_epi32(0x7f800000);
    auto indices = _mm_setr_epi32(0, 1, 2, 3);
    constexpr int population[16] = {0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4};
    for (; id <= count - 4; id += 4) {
        const auto values = _mm_loadu_ps(logits.data() + id);
        const auto bad =
            _mm_cmpeq_epi32(_mm_and_si128(_mm_castps_si128(values), exponent), exponent);
        stats.logits_examined += 4;
        if (_mm_movemask_ps(_mm_castsi128_ps(bad)))
            return false;
        const auto before = _mm_castsi128_ps(_mm_cmpgt_epi32(candidate4, indices));
        const auto wins = _mm_or_ps(_mm_cmpgt_ps(values, value4),
                                    _mm_and_ps(_mm_cmpeq_ps(values, value4), before));
        better += population[_mm_movemask_ps(wins)];
        if (better >= cutoff)
            return false;
        indices = _mm_add_epi32(indices, _mm_set1_epi32(4));
    }
#elif !defined(TOKENOMICS_LOGIT_COPY_SCALAR) && defined(__aarch64__)
    const auto value4 = vdupq_n_f32(value);
    const auto candidate4 = vdupq_n_u32(uint32_t(candidate));
    const auto exponent = vdupq_n_u32(0x7f800000);
    const uint32_t initial[] = {0, 1, 2, 3};
    auto indices = vld1q_u32(initial);
    for (; id <= count - 4; id += 4) {
        const auto values = vld1q_f32(logits.data() + id);
        const auto bad = vceqq_u32(vandq_u32(vreinterpretq_u32_f32(values), exponent), exponent);
        stats.logits_examined += 4;
        if (vmaxvq_u32(bad))
            return false;
        const auto wins =
            vorrq_u32(vcgtq_f32(values, value4),
                      vandq_u32(vceqq_f32(values, value4), vcltq_u32(indices, candidate4)));
        better += int(vaddvq_u32(vshrq_n_u32(wins, 31)));
        if (better >= cutoff)
            return false;
        indices = vaddq_u32(indices, vdupq_n_u32(4));
    }
#endif
    for (; id < count; ++id) {
        ++stats.logits_examined;
        const float other = logits[id];
        if (!std::isfinite(other))
            return false;
        if (other > value || (other == value && id < candidate))
            if (++better >= cutoff)
                return false;
    }
    return true;
}
} // namespace
const char *logit_copy_backend() {
#if !defined(TOKENOMICS_LOGIT_COPY_SCALAR) && defined(__SSE2__)
    return "sse2";
#elif !defined(TOKENOMICS_LOGIT_COPY_SCALAR) && defined(__aarch64__)
    return "neon";
#else
    return "scalar";
#endif
}
void logit_copy_draft(std::span<const int> history, int pending, int cap,
                      std::span<const float> prior_logits, std::vector<int> &out,
                      LogitCopyStats &stats, int minimum, int cutoff) {
    if (cap < 0 || cap > 8 || minimum < 1 || minimum > 32 || cutoff < 1 || cutoff > 64)
        throw std::invalid_argument("invalid logit-copy settings");
    out.clear();
    stats = {};
    stats.hint_available = !prior_logits.empty();
    if (cap == 0 || prior_logits.empty())
        return;
    if (prior_logits.size() > size_t(std::numeric_limits<int>::max()))
        throw std::invalid_argument("logit-copy vocabulary exceeds int32 IDs");
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
    struct CachedMembership {
        int id;
        bool member;
    };
    std::array<CachedMembership, 33> tested{};
    size_t tested_count = 0;
    for (int length = 32; length >= minimum; --length) {
        const auto candidate = best[length];
        if (candidate.id < 0)
            continue;
        auto cached = std::find_if(tested.begin(), tested.begin() + tested_count,
                                   [&](const auto &entry) { return entry.id == candidate.id; });
        bool member;
        if (cached == tested.begin() + tested_count) {
            member = top_member(prior_logits, candidate.id, cutoff, stats);
            tested[tested_count++] = {candidate.id, member};
        } else {
            member = cached->member;
        }
        if (!member)
            continue;
        stats.selected_match_length = length;
        for (size_t i = candidate.boundary; i < std::min(size, candidate.boundary + size_t(cap));
             ++i)
            out.push_back(token(i));
        return;
    }
}
} // namespace tokenomics
