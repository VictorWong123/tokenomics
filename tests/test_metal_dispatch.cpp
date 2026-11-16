#include "../src/metal_interleave.h"
#include "ggml-metal-common.h"
#include "ggml.h"
#include <iostream>
#include <stdexcept>
#include <vector>

static void check(bool ok) {
    if (!ok)
        throw std::runtime_error("Metal short-MV dispatch predicate mismatch");
}

int main() {
    auto *ctx = ggml_init({8 * 1024 * 1024, nullptr, true});
    check(ctx != nullptr);
    int cases = 0;
    for (auto type : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q4_0, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K,
                      GGML_TYPE_Q5_0, GGML_TYPE_Q8_0}) {
        for (int heads : {1, 4}) {
            auto *weight = ggml_new_tensor_3d(ctx, type, 256, 256, heads);
            for (int rows = 1; rows <= 32; ++rows) {
                auto *input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 256, rows, heads);
                auto *op = ggml_mul_mat(ctx, weight, input);
                const int minimum =
                    type == GGML_TYPE_F32
                        ? 6
                        : (type == GGML_TYPE_F16 || type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_0
                               ? 3
                               : 2);
                check(!tokenomics_metal_short_mv(op) && !tokenomics_metal_interleave(op));
                check(ggml_metal_op_mul_mat_use_mma(op) == (rows >= minimum && rows <= 16));
                check(ggml_metal_op_mul_mat_use_mm(op, true) == (rows > 8));
                op->flags |= tokenomics::metal_verification_graph_flag;
                check(tokenomics_metal_interleave(op) ==
                      (heads == 1 && rows >= 2 && rows <= 9 &&
                       (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q6_K ||
                        type == GGML_TYPE_Q5_0 || type == GGML_TYPE_Q8_0)));
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
    static_assert(tokenomics::metal_verification_graph(true, true, true, 3, 3));
    static_assert(!tokenomics::metal_verification_graph(true, true, true, 3, 1));
    static_assert(!tokenomics::metal_verification_graph(true, true, true, 1, 1));
    static_assert(!tokenomics::metal_verification_graph(false, true, true, 3, 3));
    static_assert(!tokenomics::metal_verification_graph(true, false, true, 3, 3));
    static_assert(!tokenomics::metal_verification_graph(true, true, false, 3, 3));
    auto *weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 8);
    auto *input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 4);
    auto *product = ggml_mul_mat(ctx, weight, input);
    auto *softmax = ggml_soft_max(ctx, product);
    auto *normal = ggml_norm(ctx, softmax, 1e-5f);
    ggml_set_output(normal);
    auto *graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, normal);
    std::vector<int32_t> original;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i)
        original.push_back(ggml_graph_node(graph, i)->flags);
    tokenomics::mark_metal_verification_graph(graph, true);
    check(tokenomics::metal_verification_graph_op(product));
    check(tokenomics::metal_verification_graph_op(softmax));
    check(!tokenomics::metal_verification_graph_op(normal));
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i)
        check((ggml_graph_node(graph, i)->flags & ~tokenomics::metal_verification_graph_flag) ==
              original[i]);
    tokenomics::mark_metal_verification_graph(graph, false);
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i)
        check(ggml_graph_node(graph, i)->flags == original[i]);
    ggml_free(ctx);
    std::cout << cases << " host dispatch cases passed (no GPU execution)\n";
}
