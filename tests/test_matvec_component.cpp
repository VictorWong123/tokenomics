// Synthetic matrix component diagnostic; not learned-model or generation speed.
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "nlohmann/json.hpp"
#ifdef TOKENOMICS_HAS_METAL
#include "ggml-metal.h"
#endif
#ifdef TOKENOMICS_METAL_SHORT_MV
#include "../src/metal_verification_graph.h"
#endif
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using Json = nlohmann::ordered_json;
static void require(bool ok, const char *message) {
    if (!ok)
        throw std::runtime_error(message);
}
static uint64_t fingerprint(const void *data, size_t size) {
    const auto *bytes = static_cast<const uint8_t *>(data);
    uint64_t hash = 14695981039346656037ULL;
    for (size_t i = 0; i < size; ++i)
        hash = (hash ^ bytes[i]) * 1099511628211ULL;
    return hash;
}
static const char *profile() {
#ifdef TOKENOMICS_METAL_Q8_SHARED
    return "q8-shared";
#elif defined(TOKENOMICS_METAL_Q8_GROUP)
    return "q8-group";
#elif defined(TOKENOMICS_METAL_INTERLEAVE)
    return "interleaved";
#elif defined(TOKENOMICS_METAL_SHORT_MV)
    return "short-mv";
#else
    return "guarded";
#endif
}
struct Plan {
    int rows;
    ggml_tensor *input, *batched;
    ggml_cgraph *batch_graph, *serial_graph;
    std::vector<ggml_tensor *> singles;
};
static void run_shape(ggml_backend_t backend, ggml_type type, int width, int outputs, bool fixture,
                      const std::string &model_id) {
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(
        ggml_init({8 * 1024 * 1024, nullptr, true}), ggml_free);
    require(bool(ctx), "context allocation failed");
    auto *weights = ggml_new_tensor_2d(ctx.get(), type, width, outputs);
    std::vector<Plan> plans;
    for (int rows : {1, 2, 3, 5, 9}) {
        Plan plan{};
        plan.rows = rows;
        plan.input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, width, rows);
        plan.batched = ggml_mul_mat(ctx.get(), weights, plan.input);
#ifdef TOKENOMICS_METAL_SHORT_MV
        if (rows > 1)
            plan.batched->flags |= tokenomics::metal_verification_graph_flag;
#endif
        plan.batch_graph = ggml_new_graph_custom(ctx.get(), 128, false);
        ggml_build_forward_expand(plan.batch_graph, plan.batched);
        plan.serial_graph = ggml_new_graph_custom(ctx.get(), 128, false);
        for (int row = 0; row < rows; ++row) {
            auto *single = ggml_view_2d(ctx.get(), plan.input, width, 1, width * sizeof(float),
                                        row * width * sizeof(float));
            auto *result = ggml_mul_mat(ctx.get(), weights, single);
            plan.singles.push_back(result);
            ggml_build_forward_expand(plan.serial_graph, result);
        }
        plans.push_back(plan);
    }
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
        ggml_backend_alloc_ctx_tensors(ctx.get(), backend), ggml_backend_buffer_free);
    require(bool(buffer), "backend buffer allocation failed");
    std::vector<uint64_t> packed((ggml_nbytes(weights) + 7) / 8);
    std::mt19937 rng(20261116 + width + outputs + int(type));
    const auto random_value = [&] { return float(int(rng() % 65536) - 32768) / 32768.0f; };
    {
        // Release the large float copy before any measured work begins.
        std::vector<float> dense(size_t(width) * outputs);
        for (auto &value : dense)
            value = random_value();
        require(ggml_quantize_chunk(type, dense.data(), packed.data(), 0, outputs, width,
                                    nullptr) == ggml_nbytes(weights),
                "quantization failed");
    }
    ggml_backend_tensor_set(weights, packed.data(), 0, ggml_nbytes(weights));
    Json identity = {{"profile", profile()},
                     {"model_id", model_id},
                     {"backend", ggml_backend_name(backend)},
                     {"type", ggml_type_name(type)},
                     {"width", width},
                     {"outputs", outputs},
                     {"weight_bytes", ggml_nbytes(weights)},
                     {"weight_fnv1a64", fingerprint(packed.data(), ggml_nbytes(weights))},
                     {"synthetic_fixture", fixture}};
    std::vector<float> activations(size_t(width) * 9);
    for (auto &value : activations)
        value = random_value();
    for (auto &plan : plans)
        ggml_backend_tensor_set(plan.input, activations.data(), 0, ggml_nbytes(plan.input));
    const auto execute = [&](ggml_cgraph *graph) {
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS,
                "backend graph computation failed");
        ggml_backend_synchronize(backend);
    };
    // Gate every shape/row count before recording any timings for this shape.
    for (auto &plan : plans) {
        execute(plan.batch_graph);
        execute(plan.serial_graph);
        std::vector<float> batch(size_t(outputs) * plan.rows), single(outputs);
        ggml_backend_tensor_get(plan.batched, batch.data(), 0, ggml_nbytes(plan.batched));
        size_t nonzero = 0;
        for (int row = 0; row < plan.rows; ++row) {
            ggml_backend_tensor_get(plan.singles[row], single.data(), 0, outputs * sizeof(float));
            for (int col = 0; col < outputs; ++col) {
                const float a = batch[row * outputs + col], b = single[col];
                if (!std::isfinite(a) || !std::isfinite(b) ||
                    std::bit_cast<uint32_t>(a) != std::bit_cast<uint32_t>(b)) {
                    auto failure = identity;
                    failure.update({{"state", "bitwise_mismatch"},
                                    {"rows", plan.rows},
                                    {"row", row},
                                    {"column", col},
                                    {"batched", a},
                                    {"serial", b}});
                    std::cout << failure.dump() << std::endl;
                    throw std::runtime_error("component failed one-row arithmetic gate");
                }
                nonzero += a != 0;
            }
        }
        require(nonzero > batch.size() / 2, "degenerate component output");
        auto record = identity;
        record.update(
            {{"state", "bitwise_passed"},
             {"rows", plan.rows},
             {"compared_values", batch.size()},
             {"input_fnv1a64", fingerprint(activations.data(), ggml_nbytes(plan.input))},
             {"output_fnv1a64", fingerprint(batch.data(), batch.size() * sizeof(float))}});
        std::cout << record.dump() << std::endl;
    }
    if (fixture)
        return;
    constexpr int repetitions = 8;
    for (auto &plan : plans) {
        for (int warmup = 0; warmup < 4; ++warmup) {
            execute(plan.batch_graph);
            execute(plan.serial_graph);
        }
        const auto measure = [&](ggml_cgraph *graph) {
            const auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < repetitions; ++i)
                execute(graph);
            return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() -
                                                             start)
                       .count() /
                   repetitions;
        };
        for (int trial = 0; trial < 16; ++trial) {
            double batch_us, serial_us;
            if (trial % 2 == 0) {
                batch_us = measure(plan.batch_graph);
                serial_us = measure(plan.serial_graph);
            } else {
                serial_us = measure(plan.serial_graph);
                batch_us = measure(plan.batch_graph);
            }
            auto record = identity;
            record.update({{"state", "timing"},
                           {"rows", plan.rows},
                           {"trial", trial},
                           {"repetitions", repetitions},
                           {"batch_us", batch_us},
                           {"serial_graph_us", serial_us}});
            std::cout << record.dump() << std::endl;
        }
    }
}
int main(int argc, char **argv) {
    try {
        require(
            argc == 2 || argc == 4,
            "usage: test_matvec_component --cpu-fixture | --metal | --shapes [--model-id MODEL]");
        const std::string mode = argv[1];
        const bool fixture = mode == "--cpu-fixture";
        require(fixture || mode == "--metal" || mode == "--shapes", "unknown component mode");
        const std::string model_id = argc == 4 ? argv[3] : "qwen25-05";
        require(argc == 2 || std::string(argv[2]) == "--model-id", "expected --model-id");
        require(model_id == "qwen25-05" || model_id == "qwen25-15", "unknown model geometry");
        struct Shape {
            ggml_type type;
            int width, outputs;
        };
        const std::vector<Shape> shapes = model_id == "qwen25-05"
                                              ? std::vector<Shape>{{GGML_TYPE_Q8_0, 896, 151936},
                                                                   {GGML_TYPE_Q5_0, 896, 896},
                                                                   {GGML_TYPE_Q4_K, 4864, 896},
                                                                   {GGML_TYPE_Q6_K, 4864, 896}}
                                              : std::vector<Shape>{{GGML_TYPE_Q6_K, 1536, 151936},
                                                                   {GGML_TYPE_Q4_K, 1536, 1536},
                                                                   {GGML_TYPE_Q6_K, 8960, 1536}};
        if (mode == "--shapes") {
            for (const auto &shape : shapes)
                std::cout << Json{{"model_id", model_id},
                                  {"type", ggml_type_name(shape.type)},
                                  {"width", shape.width},
                                  {"outputs", shape.outputs},
                                  {"weight_bytes", shape.width / ggml_blck_size(shape.type) *
                                                       ggml_type_size(shape.type) * shape.outputs}}
                                 .dump()
                          << std::endl;
            return 0;
        }
        ggml_backend_t raw = nullptr;
        if (fixture)
            raw = ggml_backend_cpu_init();
#ifdef TOKENOMICS_HAS_METAL
        else
            raw = ggml_backend_metal_init();
#endif
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(raw, ggml_backend_free);
        require(bool(backend), "requested backend unavailable; CPU fallback prohibited");
        if (fixture)
            ggml_backend_cpu_set_n_threads(backend.get(), 2);
        if (fixture) {
            for (auto type : {GGML_TYPE_Q8_0, GGML_TYPE_Q5_0, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K}) {
                if (model_id == "qwen25-15" && (type == GGML_TYPE_Q8_0 || type == GGML_TYPE_Q5_0))
                    continue;
                run_shape(backend.get(), type, 256, 128, true, model_id);
            }
        } else {
            for (const auto &shape : shapes)
                run_shape(backend.get(), shape.type, shape.width, shape.outputs, false, model_id);
        }
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
