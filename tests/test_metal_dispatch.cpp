#include "ggml-metal-common.h"
#include "ggml.h"
#include <iostream>
#include <stdexcept>

static void check(bool ok) {
    if (!ok)
        throw std::runtime_error("Metal short-MV dispatch predicate mismatch");
}

int main() {
    auto *ctx = ggml_init({8 * 1024 * 1024, nullptr, true});
    check(ctx != nullptr);
    int cases = 0;
    for (auto type :
         {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q4_0, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K}) {
        for (int heads : {1, 4}) {
            auto *weight = ggml_new_tensor_3d(ctx, type, 256, 256, heads);
            for (int rows = 1; rows <= 32; ++rows) {
                auto *input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 256, rows, heads);
                auto *op = ggml_mul_mat(ctx, weight, input);
                // Ordinary decode is still MV; 2..9 no longer select MMA/MM.
                check(ggml_metal_op_mul_mat_use_mma(op) == (rows >= 10 && rows <= 16));
                check(ggml_metal_op_mul_mat_use_mm(op, true) == (rows >= 10));
                check(!ggml_metal_op_mul_mat_use_mm(op, false));
                // Structural fusion planning stays shape-independent; runtime
                // eligibility above prevents MMA+ADD fusion for short graphs.
                check(ggml_metal_op_mul_mat_may_use_mma(op));
                ++cases;
            }
        }
    }
    ggml_free(ctx);
    std::cout << cases << " host dispatch cases passed (no GPU execution)\n";
}
