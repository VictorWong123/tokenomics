// Execute the generated Metal verification split on a CPU backend. This tests
// cache lifecycle and output-row mapping, not Metal kernel arithmetic.
#include "../src/metal_verification_graph.h"
#include "ggml-backend.h"
#include "llama.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

static void require(bool ok, const char *message) {
    if (!ok)
        throw std::runtime_error(message);
}
struct ScopeProbe {
    bool all_outputs = false, invalid = false;
    int tokens = 0, tagged = 0, untagged = 0;
};
static bool graph_scope_probe(ggml_tensor *tensor, bool ask, void *user) {
    auto &scope = *static_cast<ScopeProbe *>(user);
    if (ask && (tensor->op == GGML_OP_MUL_MAT || tensor->op == GGML_OP_SOFT_MAX)) {
        if (tokenomics::metal_verification_graph_op(tensor)) {
            ++scope.tagged;
            scope.invalid |= !scope.all_outputs || scope.tokens < 2 || scope.tokens > 9;
        } else {
            ++scope.untagged;
        }
    }
    return false; // Metadata only: no tensor readback or extra synchronization.
}
static void decode(llama_context *ctx, int first, int count, bool outputs, ScopeProbe &scope) {
    scope.all_outputs = outputs;
    scope.tokens = count;
    auto batch = llama_batch_init(count, 0, 1);
    batch.n_tokens = count;
    for (int i = 0; i < count; ++i) {
        batch.token[i] = 100 + (first + i) % 9;
        batch.pos[i] = first + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = outputs || i == count - 1;
    }
    const int status = llama_decode(ctx, batch);
    llama_batch_free(batch);
    require(status == 0, "CPU split-probe decode failed");
    require(!scope.invalid, "verification metadata leaked into prefill or ordinary graph");
    llama_synchronize(ctx);
}
static void prefill(llama_context *ctx, int count, ScopeProbe &scope) {
    llama_memory_clear(llama_get_memory(ctx), true);
    for (int offset = 0; offset < count; offset += 512)
        decode(ctx, offset, std::min(512, count - offset), false, scope);
}

int main(int argc, char **argv) {
    try {
        require(argc == 2, "usage: test_metal_context_cpu SYNTHETIC_MODEL.gguf");
        ggml_backend_load_all();
        llama_backend_init();
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(argv[1], mp), llama_model_free);
        require(bool(model), "cannot load synthetic context-probe model");
        ScopeProbe scope;
        auto cp = llama_context_default_params();
        cp.cb_eval = graph_scope_probe;
        cp.cb_eval_user_data = &scope;
        cp.n_ctx = 1024;
        cp.n_batch = cp.n_ubatch = 512;
        cp.n_outputs_max = cp.n_outputs_max_per_seq = 9;
        cp.n_threads = cp.n_threads_batch = 1;
        cp.n_seq_max = 1;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cp.offload_kqv = true; // Deliberately select split logic; weights/backend stay CPU.
        cp.op_offload = false;
        std::unique_ptr<llama_context, decltype(&llama_free)> ctx(
            llama_init_from_model(model.get(), cp), llama_free);
        require(bool(ctx), "cannot initialize CPU split probe");
        const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
        require(vocab > 109, "probe requires synthetic byte-tokenizer fixture");
        int cases = 0, comparisons = 0, nonzero = 0;
        for (int prefix :
             {3, 4, 9, 10, 248, 249, 254, 255, 256, 257, 504, 510, 511, 512, 515, 521, 522})
            for (int rows : {2, 3, 5, 9}) {
                prefill(ctx.get(), prefix, scope);
                std::vector<float> reference(size_t(vocab) * (rows + 1));
                for (int row = 0; row < rows + 1; ++row) {
                    decode(ctx.get(), prefix + row, 1, true, scope);
                    const auto *logits = llama_get_logits_ith(ctx.get(), 0);
                    require(logits != nullptr, "missing ordinary probe logits");
                    std::copy_n(logits, vocab, reference.data() + row * vocab);
                }
                prefill(ctx.get(), prefix, scope);
                decode(ctx.get(), prefix, rows, true, scope);
                const auto compare_rows = [&](int shift) {
                    for (int row = 0; row < rows; ++row) {
                        const auto *logits = llama_get_logits_ith(ctx.get(), row);
                        require(logits != nullptr, "missing split probe logits");
                        for (int id = 0; id < vocab; ++id) {
                            const float expected = reference[(row + shift) * vocab + id];
                            if (!std::isfinite(logits[id]) ||
                                std::bit_cast<uint32_t>(logits[id]) !=
                                    std::bit_cast<uint32_t>(expected)) {
                                std::cerr << "split logit mismatch prefix=" << prefix
                                          << " rows=" << rows << " row=" << row << " id=" << id
                                          << '\n';
                                throw std::runtime_error(
                                    "split logits differ from ordinary CPU logits");
                            }
                            nonzero += logits[id] != 0;
                            ++comparisons;
                        }
                    }
                };
                compare_rows(0);
                require(llama_memory_seq_pos_max(llama_get_memory(ctx.get()), 0) + 1 ==
                            prefix + rows,
                        "split probe lost cached rows");
                // Reject all but the first speculative input row, then verify
                // again from the retained prefix. This exercises physical slot
                // reuse and the cached-prefix argument after rollback.
                require(llama_memory_seq_rm(llama_get_memory(ctx.get()), 0, prefix + 1, -1),
                        "split probe rollback failed");
                require(llama_memory_seq_pos_max(llama_get_memory(ctx.get()), 0) + 1 == prefix + 1,
                        "rollback retained wrong prefix");
                decode(ctx.get(), prefix + 1, rows, true, scope);
                compare_rows(1);
                require(llama_memory_seq_pos_max(llama_get_memory(ctx.get()), 0) + 1 ==
                            prefix + 1 + rows,
                        "split probe lost rows after rollback");
                ++cases;
            }
        require(scope.tagged > 0 && scope.untagged > 0, "graph scope coverage is degenerate");
        std::cout << "graph_scope_tagged=" << scope.tagged << " untagged=" << scope.untagged
                  << '\n';
        require(nonzero >= cases, "degenerate split probe outputs");
        std::cout << cases << " CPU context/rollback cases; " << comparisons
                  << " bitwise logit comparisons passed across KV padding boundaries\n";
        ctx.reset();
        model.reset();
        llama_backend_free();
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
