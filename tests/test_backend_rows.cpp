// Direct backend arithmetic check; CPU execution does not validate Metal.
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
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
        int softmax_cases = 0, softmax_values = 0;
        for (int width : {256, 512, 768, 2048})
            for (int rows : {2, 3, 9})
                for (int heads : {1, 2, 4, 14}) {
                    softmax_values += run_softmax(backend.get(), width, rows, heads);
                    ++softmax_cases;
                }
        std::cout << softmax_cases << " masked softmax cases; " << softmax_values
                  << " bitwise comparisons passed\n";
        std::cout << cases << " backend matrix cases; " << values
                  << " bitwise row comparisons passed\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
