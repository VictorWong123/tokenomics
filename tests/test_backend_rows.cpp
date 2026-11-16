// Direct backend arithmetic check; CPU execution does not validate Metal.
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#ifdef TOKENOMICS_METAL_SHORT_MV
#include "../src/metal_verification_graph.h"
#endif
#ifdef TOKENOMICS_HAS_METAL
#include "ggml-metal.h"
#endif
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

static void check(bool condition, const char *message) {
    if (!condition)
        throw std::runtime_error(message);
}

static int run_case(ggml_backend_t backend, ggml_type type, int width, int rows, int heads,
                    int weight_heads) {
    constexpr int outputs = 128;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(
        ggml_init({8 * 1024 * 1024, nullptr, true}), ggml_free);
    check(bool(ctx), "context allocation failed");
    auto *weights = ggml_new_tensor_3d(ctx.get(), type, width, outputs, weight_heads);
    auto *input = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, width, rows, heads);
    auto *batched = ggml_mul_mat(ctx.get(), weights, input);
#ifdef TOKENOMICS_METAL_SHORT_MV
    batched->flags |= tokenomics::metal_verification_graph_flag;
#endif
    auto *graph = ggml_new_graph_custom(ctx.get(), 1024, false);
    ggml_build_forward_expand(graph, batched);
    std::vector<ggml_tensor *> singles, results;
    for (int row = 0; row < rows; ++row) {
        auto *single = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, width, 1, heads);
        auto *result = ggml_mul_mat(ctx.get(), weights, single);
        singles.push_back(single);
        results.push_back(result);
        ggml_build_forward_expand(graph, result);
    }
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
        ggml_backend_alloc_ctx_tensors(ctx.get(), backend), ggml_backend_buffer_free);
    check(bool(buffer), "backend buffer allocation failed");
    std::mt19937 rng(20261116 + width + rows + heads);
    std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);
    std::vector<float> dense(size_t(width) * outputs * weight_heads),
        activations(size_t(width) * rows * heads);
    for (auto &value : dense)
        value = uniform(rng);
    for (auto &value : activations)
        value = uniform(rng);
    // Quantize once; single and batched operations use the identical weights.
    std::vector<uint64_t> packed((ggml_nbytes(weights) + 7) / 8);
    if (type == GGML_TYPE_F32)
        std::memcpy(packed.data(), dense.data(), dense.size() * sizeof(float));
    else if (type == GGML_TYPE_F16)
        ggml_fp32_to_fp16_row(dense.data(), reinterpret_cast<ggml_fp16_t *>(packed.data()),
                              dense.size());
    else
        check(ggml_quantize_chunk(type, dense.data(), packed.data(), 0, outputs * weight_heads,
                                  width, nullptr) == ggml_nbytes(weights),
              "weight quantization failed");
    ggml_backend_tensor_set(weights, packed.data(), 0, ggml_nbytes(weights));
    ggml_backend_tensor_set(input, activations.data(), 0, ggml_nbytes(input));
    for (int row = 0; row < rows; ++row) {
        std::vector<float> single(size_t(width) * heads);
        for (int head = 0; head < heads; ++head)
            std::copy_n(activations.data() + (head * rows + row) * width, width,
                        single.data() + head * width);
        ggml_backend_tensor_set(singles[row], single.data(), 0, ggml_nbytes(singles[row]));
    }
    check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS,
          "backend computation failed");
    ggml_backend_synchronize(backend);
    std::vector<float> batch_result(size_t(outputs) * rows * heads),
        single_result(size_t(outputs) * heads);
    ggml_backend_tensor_get(batched, batch_result.data(), 0, ggml_nbytes(batched));
    int nonzero = 0;
    for (int row = 0; row < rows; ++row) {
        ggml_backend_tensor_get(results[row], single_result.data(), 0, ggml_nbytes(results[row]));
        for (int head = 0; head < heads; ++head)
            for (int column = 0; column < outputs; ++column) {
                const float a = batch_result[(head * rows + row) * outputs + column];
                const float b = single_result[head * outputs + column];
                if (!std::isfinite(a) || !std::isfinite(b) ||
                    std::bit_cast<uint32_t>(a) != std::bit_cast<uint32_t>(b)) {
                    std::cerr << "row mismatch type=" << ggml_type_name(type) << " width=" << width
                              << " rows=" << rows << " heads=" << heads
                              << " weight_heads=" << weight_heads << " row=" << row
                              << " column=" << column << " batched=" << a << " single=" << b
                              << '\n';
                    throw std::runtime_error("backend rows are not bitwise invariant");
                }
                nonzero += a != 0;
            }
    }
    check(nonzero > outputs * rows * heads / 2, "degenerate matrix result");
    return outputs * rows * heads;
}

static int run_softmax(ggml_backend_t backend, int width, int rows, int heads) {
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(
        ggml_init({8 * 1024 * 1024, nullptr, true}), ggml_free);
    check(bool(ctx), "softmax context allocation failed");
    auto *input = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, width, rows, heads);
    auto *mask = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, width, rows);
    auto *batch = ggml_soft_max_ext(ctx.get(), input, mask, 0.125f, 0.0f);
#ifdef TOKENOMICS_METAL_SHORT_MV
    batch->flags |= tokenomics::metal_verification_graph_flag;
#endif
    auto *graph = ggml_new_graph_custom(ctx.get(), 1024, false);
    ggml_build_forward_expand(graph, batch);
    std::vector<ggml_tensor *> inputs, masks, results;
    for (int row = 0; row < rows; ++row) {
        inputs.push_back(ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, width, 1, heads));
        masks.push_back(ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, width, 1));
        results.push_back(ggml_soft_max_ext(ctx.get(), inputs.back(), masks.back(), 0.125f, 0.0f));
        ggml_build_forward_expand(graph, results.back());
    }
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
        ggml_backend_alloc_ctx_tensors(ctx.get(), backend), ggml_backend_buffer_free);
    check(bool(buffer), "softmax backend allocation failed");
    std::mt19937 rng(12345 + width + rows + heads);
    std::uniform_real_distribution<float> uniform(-8.0f, 8.0f);
    std::vector<float> values(size_t(width) * rows * heads), masking(size_t(width) * rows);
    for (auto &value : values)
        value = uniform(rng);
    for (int row = 0; row < rows; ++row)
        for (int col = 0; col < width; ++col)
            masking[row * width + col] = col < std::min(width, 249 + row) ? 0.0f : -INFINITY;
    ggml_backend_tensor_set(input, values.data(), 0, ggml_nbytes(input));
    ggml_backend_tensor_set(mask, masking.data(), 0, ggml_nbytes(mask));
    for (int row = 0; row < rows; ++row) {
        std::vector<float> single(size_t(width) * heads);
        for (int head = 0; head < heads; ++head)
            std::copy_n(values.data() + (head * rows + row) * width, width,
                        single.data() + head * width);
        ggml_backend_tensor_set(inputs[row], single.data(), 0, ggml_nbytes(inputs[row]));
        ggml_backend_tensor_set(masks[row], masking.data() + row * width, 0,
                                ggml_nbytes(masks[row]));
    }
    check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS,
          "softmax compute failed");
    ggml_backend_synchronize(backend);
    std::vector<float> batched(values.size()), single(size_t(width) * heads);
    ggml_backend_tensor_get(batch, batched.data(), 0, ggml_nbytes(batch));
    for (int row = 0; row < rows; ++row) {
        ggml_backend_tensor_get(results[row], single.data(), 0, ggml_nbytes(results[row]));
        for (int head = 0; head < heads; ++head) {
            double sum = 0;
            for (int col = 0; col < width; ++col) {
                const float a = batched[(head * rows + row) * width + col];
                const float b = single[head * width + col];
                if (!std::isfinite(a) || !std::isfinite(b) ||
                    std::bit_cast<uint32_t>(a) != std::bit_cast<uint32_t>(b)) {
                    std::cerr << "softmax mismatch width=" << width << " rows=" << rows
                              << " heads=" << heads << " row=" << row << " col=" << col << '\n';
                    throw std::runtime_error("softmax rows are not bitwise invariant");
                }
                sum += a;
            }
            check(std::abs(sum - 1.0) < 1e-5, "invalid softmax probability sum");
        }
    }
    return int(values.size());
}

// Normalization and elementwise kernels are deliberately unmarked: the owned
// graph selector marks only matmul/softmax. Test their ordinary dispatch too.
static int compare_row_outputs(ggml_backend_t backend, ggml_tensor *batch,
                               const std::vector<ggml_tensor *> &singles, int row_width,
                               const char *operation) {
    std::vector<float> all(size_t(row_width) * singles.size()), one(row_width);
    ggml_backend_tensor_get(batch, all.data(), 0, ggml_nbytes(batch));
    size_t nonzero = 0;
    for (size_t row = 0; row < singles.size(); ++row) {
        ggml_backend_tensor_get(singles[row], one.data(), 0, ggml_nbytes(singles[row]));
        for (int col = 0; col < row_width; ++col) {
            const float a = all[row * row_width + col], b = one[col];
            if (!std::isfinite(a) || !std::isfinite(b) ||
                std::bit_cast<uint32_t>(a) != std::bit_cast<uint32_t>(b)) {
                std::cerr << operation << " mismatch row_width=" << row_width
                          << " rows=" << singles.size() << " row=" << row << " col=" << col
                          << " batched=" << a << " single=" << b << '\n';
                throw std::runtime_error("non-matmul rows are not bitwise invariant");
            }
            nonzero += a != 0;
        }
    }
    check(nonzero > all.size() / 2, "degenerate non-matmul output");
    return int(all.size());
}

static int run_row_operation(ggml_backend_t backend, int width, int rows, int mode) {
    const char *names[] = {"rms_norm", "rms_norm_mul", "rms_norm_mul_add",
                           "silu",     "silu_mul",     "swiglu_split"};
    check(mode >= 0 && mode < 6, "unknown row operation");
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(
        ggml_init({8 * 1024 * 1024, nullptr, true}), ggml_free);
    check(bool(ctx), "row-operation context allocation failed");
    auto *scale = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, width);
    auto *bias = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, width);
    const auto operation = [&](ggml_tensor *input, ggml_tensor *other) {
        switch (mode) {
        case 0:
            return ggml_rms_norm(ctx.get(), input, 1e-6f);
        case 1:
            return ggml_mul(ctx.get(), ggml_rms_norm(ctx.get(), input, 1e-6f), scale);
        case 2:
            return ggml_add(ctx.get(),
                            ggml_mul(ctx.get(), ggml_rms_norm(ctx.get(), input, 1e-6f), scale),
                            bias);
        case 3:
            return ggml_silu(ctx.get(), input);
        case 4:
            return ggml_mul(ctx.get(), ggml_silu(ctx.get(), input), other);
        default:
            return ggml_swiglu_split(ctx.get(), input, other);
        }
    };
    auto *input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, width, rows);
    auto *other = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, width, rows);
    auto *batch = operation(input, other);
    auto *graph = ggml_new_graph_custom(ctx.get(), 1024, false);
    ggml_build_forward_expand(graph, batch);
    std::vector<ggml_tensor *> inputs, others, results;
    for (int row = 0; row < rows; ++row) {
        inputs.push_back(ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, width, 1));
        others.push_back(ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, width, 1));
        results.push_back(operation(inputs.back(), others.back()));
        ggml_build_forward_expand(graph, results.back());
    }
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
        ggml_backend_alloc_ctx_tensors(ctx.get(), backend), ggml_backend_buffer_free);
    check(bool(buffer), "row-operation backend allocation failed");
    std::mt19937 rng(20261116 + width + rows + mode);
    const auto fill = [&](ggml_tensor *tensor, float magnitude) {
        std::vector<float> values(ggml_nelements(tensor));
        for (auto &value : values)
            value = magnitude * float(int(rng() % 65536) - 32768) / 32768.0f;
        ggml_backend_tensor_set(tensor, values.data(), 0, ggml_nbytes(tensor));
        return values;
    };
    fill(scale, 1.0f);
    fill(bias, 1.0f);
    auto values = fill(input, 8.0f), factors = fill(other, 2.0f);
    for (int row = 0; row < rows; ++row) {
        ggml_backend_tensor_set(inputs[row], values.data() + row * width, 0,
                                ggml_nbytes(inputs[row]));
        ggml_backend_tensor_set(others[row], factors.data() + row * width, 0,
                                ggml_nbytes(others[row]));
    }
    check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS,
          "row-operation compute failed");
    ggml_backend_synchronize(backend);
    return compare_row_outputs(backend, batch, results, width, names[mode]);
}

static int run_rope(ggml_backend_t backend, int width, int rows, int heads, int prefix, int mode) {
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(
        ggml_init({8 * 1024 * 1024, nullptr, true}), ggml_free);
    check(bool(ctx), "RoPE context allocation failed");
    auto *input = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, width, heads, rows);
    auto *position = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, rows);
    const auto operation = [&](ggml_tensor *source, ggml_tensor *positions) {
        return ggml_rope_ext(ctx.get(), source, positions, nullptr, width, mode, 32768, 1000000.0f,
                             1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    };
    auto *batch = operation(input, position);
    auto *graph = ggml_new_graph_custom(ctx.get(), 1024, false);
    ggml_build_forward_expand(graph, batch);
    std::vector<ggml_tensor *> inputs, positions, results;
    for (int row = 0; row < rows; ++row) {
        inputs.push_back(ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, width, heads, 1));
        positions.push_back(ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 1));
        results.push_back(operation(inputs.back(), positions.back()));
        ggml_build_forward_expand(graph, results.back());
    }
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
        ggml_backend_alloc_ctx_tensors(ctx.get(), backend), ggml_backend_buffer_free);
    check(bool(buffer), "RoPE backend allocation failed");
    std::mt19937 rng(20261116 + width + rows + heads + prefix + mode);
    std::vector<float> values(size_t(width) * heads * rows);
    std::vector<int32_t> indices(rows);
    for (auto &value : values)
        value = float(int(rng() % 65536) - 32768) / 32768.0f;
    for (int row = 0; row < rows; ++row)
        indices[row] = prefix + row;
    ggml_backend_tensor_set(input, values.data(), 0, ggml_nbytes(input));
    ggml_backend_tensor_set(position, indices.data(), 0, ggml_nbytes(position));
    for (int row = 0; row < rows; ++row) {
        ggml_backend_tensor_set(inputs[row], values.data() + size_t(row) * width * heads, 0,
                                ggml_nbytes(inputs[row]));
        ggml_backend_tensor_set(positions[row], &indices[row], 0, sizeof(int32_t));
    }
    check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "RoPE compute failed");
    ggml_backend_synchronize(backend);
    return compare_row_outputs(backend, batch, results, width * heads,
                               mode == GGML_ROPE_TYPE_NEOX ? "RoPE-NEOX" : "RoPE-normal");
}

int main(int argc, char **argv) {
    try {
        check(argc == 2 && (std::string(argv[1]) == "--cpu" || std::string(argv[1]) == "--metal" ||
                            std::string(argv[1]) == "--metal-compile-only"),
              "usage: test_backend_rows --cpu|--metal|--metal-compile-only");
        ggml_backend_t raw = nullptr;
        if (std::string(argv[1]) != "--cpu") {
#ifdef TOKENOMICS_HAS_METAL
            raw = ggml_backend_metal_init();
            check(raw && ggml_backend_is_metal(raw), "requested Metal backend unavailable");
#else
            throw std::runtime_error("Metal was not compiled; CPU fallback is forbidden");
#endif
        } else {
            raw = ggml_backend_cpu_init();
            check(raw != nullptr, "CPU backend unavailable");
            ggml_backend_cpu_set_n_threads(raw, 1);
        }
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(raw, ggml_backend_free);
        check(bool(backend), "requested backend unavailable");
        std::cout << "backend=" << ggml_backend_name(backend.get()) << '\n';
        if (std::string(argv[1]) == "--metal-compile-only") {
            std::cout << "Metal libraries compiled; GPU graph execution skipped\n";
            return 0;
        }
        int cases = 0, values = 0;
        for (auto type : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K,
                          GGML_TYPE_Q5_0, GGML_TYPE_Q8_0})
            for (int width : {64, 128, 256, 896, 1536, 4864})
                for (int rows : {2, 3, 5, 7, 9})
                    for (int heads : {1, 4}) {
                        // Quantized widths must align with their 256-value blocks.
                        if (width % ggml_blck_size(type))
                            continue;
                        for (int weight_heads : {1, 4}) {
                            if (weight_heads > heads)
                                continue;
                            values +=
                                run_case(backend.get(), type, width, rows, heads, weight_heads);
                            ++cases;
                        }
                    }
        // Actual Qwen0.5B grouped-query attention has 14 query heads and
        // 2 KV heads: exercise the odd broadcast ratio, including KV widths.
        for (auto type : {GGML_TYPE_F32, GGML_TYPE_F16})
            for (int width : {64, 256, 512, 768, 2048})
                for (int rows : {2, 3, 5, 7, 9}) {
                    values += run_case(backend.get(), type, width, rows, 14, 2);
                    ++cases;
                }
        int softmax_cases = 0, softmax_values = 0;
        for (int width : {256, 512, 768, 2048})
            for (int rows : {2, 3, 9})
                for (int heads : {1, 2, 4, 14}) {
                    softmax_values += run_softmax(backend.get(), width, rows, heads);
                    ++softmax_cases;
                }
        int row_cases = 0, row_values = 0;
        for (int width : {32, 896, 1536, 4864})
            for (int rows : {2, 3, 5, 7, 9})
                for (int mode = 0; mode < 3; ++mode) {
                    row_values += run_row_operation(backend.get(), width, rows, mode);
                    ++row_cases;
                }
        // 4864-wide FFN activations cross the unary contiguous-specialization
        // threshold of32768 elements between six and seven query rows.
        for (int width : {896, 4864})
            for (int rows : {2, 3, 5, 6, 7, 9})
                for (int mode = 3; mode < 6; ++mode) {
                    row_values += run_row_operation(backend.get(), width, rows, mode);
                    ++row_cases;
                }
        int rope_cases = 0, rope_values = 0;
        for (int width : {64, 128})
            for (int rows : {2, 3, 5, 7, 9})
                for (int heads : {2, 14})
                    for (int prefix : {0, 248, 255, 511})
                        for (int mode : {GGML_ROPE_TYPE_NORMAL, GGML_ROPE_TYPE_NEOX}) {
                            rope_values +=
                                run_rope(backend.get(), width, rows, heads, prefix, mode);
                            ++rope_cases;
                        }
        std::cout << row_cases << " normalization/activation cases; " << row_values
                  << " bitwise comparisons passed\n";
        std::cout << rope_cases << " positional RoPE cases; " << rope_values
                  << " bitwise comparisons passed\n";
        std::cout << softmax_cases << " masked softmax cases; " << softmax_values
                  << " bitwise comparisons passed\n";
        std::cout << cases << " backend matrix cases; " << values
                  << " bitwise row comparisons passed\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
