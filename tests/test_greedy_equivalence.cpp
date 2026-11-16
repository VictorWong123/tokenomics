#include "llama.h"
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

int main(int argc, char **argv) {
    std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> reference(
        llama_sampler_chain_init(llama_sampler_chain_default_params()), llama_sampler_free);
    llama_sampler_chain_add(reference.get(), llama_sampler_init_temp_ext(0.0f, 0.0f, 1.0f));
    llama_sampler_chain_add(reference.get(), llama_sampler_init_greedy());
    std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> candidate(
        llama_sampler_init_greedy(), llama_sampler_free);
    if (argc == 2 && std::string(argv[1]) == "--bench") {
        benchmark(reference.get(), candidate.get());
        return 0;
    }
    if (argc != 1)
        throw std::runtime_error("usage: test_greedy_equivalence [--bench]");
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
            auto copied = original;
            llama_token_data_array a{original.data(), original.size(), -1, false};
            llama_token_data_array b{copied.data(), copied.size(), -1, false};
            llama_sampler_apply(reference.get(), &a);
            llama_sampler_apply(candidate.get(), &b);
            if (a.selected != b.selected || a.data[a.selected].id != b.data[b.selected].id)
                throw std::runtime_error("temperature-zero/greedy selection mismatch");
            ++cases;
        }
    std::cout << cases
              << " exact greedy selection cases passed, including ties and nonfinite inputs\n";
}
