#pragma once
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__)
#error "Strict greedy requires IEEE NaN/infinity comparisons; disable fast-math"
#endif
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace tokenomics {
// Match the pinned host greedy sampler: earliest maximum, ignore later NaNs,
// retain token zero if the first logit is NaN. Never compile with fast-math.
inline std::int32_t strict_greedy_argmax(const float *logits, std::size_t size) {
    if (!logits || size == 0 || size > std::size_t(INT32_MAX))
        throw std::invalid_argument("strict greedy requires a nonempty int32 vocabulary");
    constexpr std::size_t lanes = 8;
    std::array<float, lanes> maxima;
    maxima.fill(-std::numeric_limits<float>::infinity());
    std::array<std::int32_t, lanes> indices{};
    std::size_t i = 0;
    for (; i + lanes <= size; i += lanes)
        for (std::size_t lane = 0; lane < lanes; ++lane)
            if (logits[i + lane] > maxima[lane]) {
                maxima[lane] = logits[i + lane];
                indices[lane] = std::int32_t(i + lane);
            }
    float best = logits[0];
    std::int32_t selected = 0;
    for (std::size_t lane = 0; lane < lanes; ++lane)
        if (maxima[lane] > best || (maxima[lane] == best && indices[lane] < selected)) {
            best = maxima[lane];
            selected = indices[lane];
        }
    for (; i < size; ++i)
        if (logits[i] > best) {
            best = logits[i];
            selected = std::int32_t(i);
        }
    return selected;
}
} // namespace tokenomics
