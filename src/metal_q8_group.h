#pragma once
#include "metal_q8_group_constants.h"
#include "metal_short_mv.h"

static_assert(TOKENOMICS_METAL_Q8_GROUP_ROWS == 2,
              "Review the named kernel before changing group size");
static inline int tokenomics_metal_q8_group_rows(const ggml_tensor *op) {
    return tokenomics_metal_short_mv(op) && op->src[0]->type == GGML_TYPE_Q8_0 &&
                   op->src[0]->ne[2] == 1 && op->src[0]->ne[3] == 1 && op->src[1]->ne[2] == 1 &&
                   op->src[1]->ne[3] == 1 && !ggml_is_transposed(op->src[0]) &&
                   !ggml_is_transposed(op->src[1])
               ? TOKENOMICS_METAL_Q8_GROUP_ROWS
               : 1;
}
