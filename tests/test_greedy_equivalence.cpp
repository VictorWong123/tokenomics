#include "../src/greedy_direct.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "llama.h"
#ifdef TOKENOMICS_HAS_METAL
#include "ggml-metal.h"
#endif
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

static void benchmark(llama_sampler *reference, llama_sampler *candidate) {
    using Clock = std::chrono::steady_clock;
    for (int size : {263, 32768, 151936}) {
        std::mt19937 rng(20261116 + size);
        std::vector<float> logits(size);
        for (auto &value : logits)
            value = float(int(rng() % 65536) - 32768) / 1024;
        std::vector<llama_token_data> data(size);
        constexpr int rounds = 32;
        auto measure = [&](llama_sampler *sampler) {
            std::uint64_t checksum = 0;
            const auto start = Clock::now();
            for (int round = 0; round < rounds; ++round) {
                // Match common_sampler::set_logits candidate preparation.
                for (int i = 0; i < size; ++i)
                    data[i] = {i, logits[i], 0.0f};
                llama_token_data_array values{data.data(), data.size(), -1, false};
                llama_sampler_apply(sampler, &values);
                checksum += values.data[values.selected].id;
            }
            const auto ns =
                std::chrono::duration<double, std::nano>(Clock::now() - start).count() / rounds;
            return std::pair{ns, checksum};
        };
        measure(reference);
        measure(candidate);
        for (int trial = 0; trial < 16; ++trial) {
            std::pair<double, std::uint64_t> a, b;
            if (trial % 2 == 0) {
                a = measure(reference);
                b = measure(candidate);
            } else {
                b = measure(candidate);
                a = measure(reference);
            }
            if (a.second != b.second)
                throw std::runtime_error("benchmark selection mismatch");
            std::cout << "{\"vocabulary\":" << size << ",\"trial\":" << trial
                      << ",\"rounds\":" << rounds << ",\"reference_ns\":" << a.first
                      << ",\"candidate_ns\":" << b.first << ",\"checksum\":" << a.second << "}\n";
        }
    }
}

// Component timing includes candidate preparation for the two existing paths.
// Direct selection reads the same logits without materializing token_data.
static void benchmark_direct(llama_sampler *reference, llama_sampler *no_temp) {
    using Clock = std::chrono::steady_clock;
    for (int size : {263, 32768, 151936}) {
        std::mt19937 rng(20261116 + size);
        std::vector<float> logits(size);
        for (auto &value : logits)
            value = float(int(rng() % 65536) - 32768) / 1024;
        std::vector<llama_token_data> data(size);
        constexpr int rounds = 32;
        auto measure = [&](int method) {
            std::uint64_t checksum = 0;
            auto start = Clock::now();
            for (int round = 0; round < rounds; ++round) {
                // Prevent the compiler from hoisting a pure repeated argmax.
                logits[round % size] += float((round & 1) ? 1 : -1) / 65536;
                if (method == 2) {
                    checksum += tokenomics::strict_greedy_argmax(logits.data(), logits.size());
                } else {
                    for (int i = 0; i < size; ++i)
                        data[i] = {i, logits[i], 0.0f};
                    llama_token_data_array values{data.data(), data.size(), -1, false};
                    llama_sampler_apply(method == 0 ? reference : no_temp, &values);
                    checksum += values.data[values.selected].id;
                }
                logits[round % size] -= float((round & 1) ? 1 : -1) / 65536;
            }
            const double ns =
                std::chrono::duration<double, std::nano>(Clock::now() - start).count() / rounds;
            return std::pair{ns, checksum};
        };
        for (int method = 0; method < 3; ++method)
            measure(method);
        for (int trial = 0; trial < 16; ++trial) {
            std::array<std::pair<double, std::uint64_t>, 3> results;
            for (int offset = 0; offset < 3; ++offset) {
                const int method = (trial + offset) % 3;
                results[method] = measure(method);
            }
            if (results[0].second != results[1].second || results[0].second != results[2].second)
                throw std::runtime_error("direct benchmark selection mismatch");
            std::cout << "{\"vocabulary\":" << size << ",\"trial\":" << trial
                      << ",\"rounds\":" << rounds << ",\"reference_ns\":" << results[0].first
                      << ",\"no_temp_ns\":" << results[1].first
                      << ",\"direct_ns\":" << results[2].first
                      << ",\"checksum\":" << results[0].second << "}\n";
        }
    }
}

// This is a negative-capability diagnostic, not a backend-sampling fast path.
// GGML argmax and the adapter's strict-first greedy sampler have different tie
// contracts at the pinned revision. Report examples without using an invalid
// index for a gather or changing inference behavior.
static void argmax_probe(llama_sampler *canonical, bool metal) {
    ggml_backend_t raw = nullptr;
    if (!metal)
        raw = ggml_backend_cpu_init();
#ifdef TOKENOMICS_HAS_METAL
    else
        raw = ggml_backend_metal_init();
#endif
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(raw, ggml_backend_free);
    if (!backend)
        throw std::runtime_error("requested probe backend unavailable; no CPU fallback");
    if (!metal)
        ggml_backend_cpu_set_n_threads(backend.get(), 1);
    int cases = 0, mismatches = 0;
    for (int size : {263, 4096, 151936}) {
        for (int kind = 0; kind < 9; ++kind) {
            const char *names[] = {
                "unique_last",           "tie_first_last",        "tie_adjacent", "tie_stride32",
                "positive_infinity_tie", "all_negative_infinity", "nan_first",    "nan_middle",
                "signed_zero_tie"};
            std::vector<float> logits(size, -2.0f);
            switch (kind) {
            case 0:
                logits.back() = 7.0f;
                break;
            case 1:
                logits.front() = logits.back() = 7.0f;
                break;
            case 2:
                logits[0] = logits[1] = 7.0f;
                break;
            case 3:
                logits[0] = logits[32] = 7.0f;
                break;
            case 4:
                logits.front() = logits.back() = INFINITY;
                break;
            case 5:
                std::fill(logits.begin(), logits.end(), -INFINITY);
                break;
            case 6:
                logits.front() = NAN;
                logits.back() = 7.0f;
                break;
            case 7:
                logits.front() = 7.0f;
                logits[size / 2] = NAN;
                break;
            case 8:
                for (int i = 0; i < size; ++i)
                    logits[i] = i % 2 ? -0.0f : 0.0f;
                break;
            }
            std::vector<llama_token_data> data(size);
            for (int i = 0; i < size; ++i)
                data[i] = {i, logits[i], 0.0f};
            llama_token_data_array values{data.data(), data.size(), -1, false};
            llama_sampler_apply(canonical, &values);
            const int expected = values.data[values.selected].id;
            std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(
                ggml_init({1024 * 1024, nullptr, true}), ggml_free);
            if (!ctx)
                throw std::runtime_error("argmax probe context allocation failed");
            auto *input = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, size);
            auto *selected = ggml_argmax(ctx.get(), input);
            if (!ggml_backend_supports_op(backend.get(), selected))
                throw std::runtime_error("argmax unavailable on the requested backend");
            auto *graph = ggml_new_graph_custom(ctx.get(), 128, false);
            ggml_build_forward_expand(graph, selected);
            std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
                ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()), ggml_backend_buffer_free);
            if (!buffer)
                throw std::runtime_error("argmax probe buffer allocation failed");
            ggml_backend_tensor_set(input, logits.data(), 0, ggml_nbytes(input));
            if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS)
                throw std::runtime_error("argmax probe computation failed");
            ggml_backend_synchronize(backend.get());
            int32_t actual = -1;
            ggml_backend_tensor_get(selected, &actual, 0, sizeof(actual));
            ++cases;
            mismatches += actual != expected;
            std::cout << "{\"backend\":\"" << ggml_backend_name(backend.get())
                      << "\",\"vocabulary\":" << size << ",\"input_kind\":\"" << names[kind]
                      << "\",\"canonical_id\":" << expected << ",\"backend_argmax_id\":" << actual
                      << ",\"matches\":" << (actual == expected ? "true" : "false") << "}\n";
        }
    }
    std::cout << "{\"state\":\"diagnostic_complete\",\"cases\":" << cases
              << ",\"selection_mismatches\":" << mismatches
              << ",\"runtime_fast_path_enabled\":false,\"performance_claim\":null}\n";
}

int main(int argc, char **argv) {
    std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> reference(
        llama_sampler_chain_init(llama_sampler_chain_default_params()), llama_sampler_free);
    llama_sampler_chain_add(reference.get(), llama_sampler_init_temp_ext(0.0f, 0.0f, 1.0f));
    llama_sampler_chain_add(reference.get(), llama_sampler_init_greedy());
    std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> candidate(
        llama_sampler_init_greedy(), llama_sampler_free);
    if (argc == 2 && (std::string(argv[1]) == "--argmax-probe-cpu" ||
                      std::string(argv[1]) == "--argmax-probe-metal")) {
        argmax_probe(reference.get(), std::string(argv[1]) == "--argmax-probe-metal");
        return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "--bench-direct") {
        benchmark_direct(reference.get(), candidate.get());
        return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "--bench") {
        benchmark(reference.get(), candidate.get());
        return 0;
    }
    if (argc != 1)
        throw std::runtime_error("usage: test_greedy_equivalence [--bench | --bench-direct | "
                                 "--argmax-probe-cpu | --argmax-probe-metal]");
    std::mt19937 rng(20261116);
    int cases = 0;
    for (int size : {1, 2, 7, 263, 4096, 151936})
        for (int trial = 0; trial < 64; ++trial) {
            std::vector<llama_token_data> original(size);
            for (int i = 0; i < size; ++i)
                original[i] = {i, float(int(rng() % 129) - 64), 0.0f};
            const float special[] = {0.0f, INFINITY, -INFINITY,
                                     std::numeric_limits<float>::quiet_NaN()};
            if (trial < 4)
                for (auto &token : original)
                    token.logit = special[trial];
            else if (trial < 20)
                original[(trial - 4) % size].logit = special[(trial - 4) / 4];
            std::vector<float> raw_logits;
            for (const auto &token : original)
                raw_logits.push_back(token.logit);
            const auto direct =
                tokenomics::strict_greedy_argmax(raw_logits.data(), raw_logits.size());
            auto copied = original;
            llama_token_data_array a{original.data(), original.size(), -1, false};
            llama_token_data_array b{copied.data(), copied.size(), -1, false};
            llama_sampler_apply(reference.get(), &a);
            llama_sampler_apply(candidate.get(), &b);
            if (a.selected != b.selected || a.data[a.selected].id != b.data[b.selected].id)
                throw std::runtime_error("temperature-zero/greedy selection mismatch");
            if (direct != a.data[a.selected].id)
                throw std::runtime_error("raw-logit strict greedy selection mismatch");
            ++cases;
        }
    // Exhaustive ties/nonfinite placements straddle the vector and tail boundary.
    int exhaustive = 0;
    for (int size = 1; size <= 9; ++size) {
        const float alphabet[] = {-INFINITY, -3.0f, INFINITY, NAN};
        for (int code = 0; code < (1 << (2 * size)); ++code) {
            std::vector<float> logits(size);
            std::vector<llama_token_data> data(size);
            int remaining = code;
            for (int i = 0; i < size; ++i) {
                logits[i] = alphabet[remaining & 3];
                remaining >>= 2;
                data[i] = {i, logits[i], 0.0f};
            }
            llama_token_data_array values{data.data(), data.size(), -1, false};
            llama_sampler_apply(reference.get(), &values);
            if (tokenomics::strict_greedy_argmax(logits.data(), logits.size()) !=
                values.data[values.selected].id)
                throw std::runtime_error("exhaustive strict greedy mismatch");
            ++exhaustive;
        }
    }
    for (int invalid = 0; invalid < 3; ++invalid) {
        const float value = 0;
        bool rejected = false;
        try {
            tokenomics::strict_greedy_argmax(invalid == 0 ? nullptr : &value,
                                             invalid == 1   ? 0
                                             : invalid == 2 ? std::size_t(INT32_MAX) + 1
                                                            : 1);
        } catch (const std::invalid_argument &) {
            rejected = true;
        }
        if (!rejected)
            throw std::runtime_error("invalid raw greedy input accepted");
    }
    std::cout << exhaustive << " exhaustive raw-logit cases passed\n";
    std::cout << cases
              << " exact greedy selection cases passed, including ties and nonfinite inputs\n";
}
