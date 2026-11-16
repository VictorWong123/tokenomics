#pragma once

#include "ggml.h"
#include <cstdint>

namespace tokenomics {
// Owned pinned-backend metadata. Upstream's allocator/scheduler tests only its
// named flag bits, and tensor copies preserve flags. No global/thread-local
// state is needed when Metal encodes a graph on concurrent worker threads.
inline constexpr std::int32_t metal_verification_graph_flag = 1 << 30;
static_assert((metal_verification_graph_flag &
               (GGML_TENSOR_FLAG_INPUT | GGML_TENSOR_FLAG_OUTPUT | GGML_TENSOR_FLAG_PARAM |
                GGML_TENSOR_FLAG_LOSS | GGML_TENSOR_FLAG_COMPUTE)) == 0);

constexpr bool metal_verification_graph(bool offload, bool causal, bool single_sequence,
                                        std::uint32_t tokens, std::uint32_t outputs) {
    return offload && causal && single_sequence && tokens >= 2 && tokens <= 9 && outputs == tokens;
}

inline bool metal_verification_graph_op(const ggml_tensor *op) {
    return (op->flags & metal_verification_graph_flag) != 0;
}

inline void mark_metal_verification_graph(ggml_cgraph *graph, bool verification) {
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        auto *node = ggml_graph_node(graph, i);
        if (node->op != GGML_OP_MUL_MAT && node->op != GGML_OP_SOFT_MAX)
            continue;
        if (verification)
            node->flags |= metal_verification_graph_flag;
        else
            node->flags &= ~metal_verification_graph_flag;
    }
}
} // namespace tokenomics
