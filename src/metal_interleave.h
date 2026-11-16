#pragma once

#include "metal_short_mv.h"

// Keep the first experiment limited to dense projection weights. The dispatch
// permutation changes scheduling locality, never the per-threadgroup reduction.
static inline bool tokenomics_metal_interleave(const ggml_tensor *op) {
    return tokenomics_metal_short_mv(op) &&
           (op->src[0]->type == GGML_TYPE_Q4_K || op->src[0]->type == GGML_TYPE_Q6_K ||
            op->src[0]->type == GGML_TYPE_Q5_0 || op->src[0]->type == GGML_TYPE_Q8_0) &&
           op->src[0]->ne[2] == 1 && op->src[0]->ne[3] == 1 && op->src[1]->ne[2] == 1 &&
           op->src[1]->ne[3] == 1;
}
