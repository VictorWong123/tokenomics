#pragma once
#include <cstdint>

namespace tokenomics {
// For contiguous full-attention sequence0, each chunk must remain inside the
// same 256-cell padded KV extent as its ordinary one-row decodes. The pinned
// cache applies each microbatch before calculating its visible padded extent.
constexpr std::uint32_t metal_verification_ubatch(std::uint32_t configured,
                                                  std::int64_t cached_prefix, std::uint32_t tokens,
                                                  std::uint32_t outputs, bool offload, bool causal,
                                                  bool single_sequence) {
    if (!offload || !causal || tokens == 0 || tokens > 9 || outputs != tokens)
        return configured;
    if (!single_sequence || cached_prefix < 0 || configured < tokens)
        return 1;
    const auto room = 256u - std::uint32_t(cached_prefix % 256);
    // If a boundary is crossed, the first chunk ends exactly at it. Since the
    // request has <=9 rows, the remaining chunks cannot cross another boundary.
    return room < tokens ? room : configured;
}
} // namespace tokenomics
