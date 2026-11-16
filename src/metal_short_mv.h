#pragma once

#include "ggml.h"
#include "metal_verification_graph.h"

// Experimental selector, not a guarantee of whole-graph batch invariance.
// Includes attention matmuls: their reduction order must also match one-row
// decode. FWHT remains upstream's responsibility and is dispatched first.
static inline bool tokenomics_metal_short_mv(const ggml_tensor *op) {
    return tokenomics::metal_verification_graph_op(op) && op->src[1]->type == GGML_TYPE_F32 &&
           op->src[1]->ne[1] >= 2 && op->src[1]->ne[1] <= 9;
}
