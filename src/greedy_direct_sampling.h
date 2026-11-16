#pragma once
#include <cstdint>
struct common_sampler;
// Private target-only adapter hooks. Never enable for a sampler whose caller
// consumes distributions, supports stochastic rejection, or uses backend IDs.
bool tokenomics_common_enable_direct(common_sampler *sampler);
bool tokenomics_common_direct_enabled(const common_sampler *sampler);
std::uint64_t tokenomics_common_direct_calls(const common_sampler *sampler);
