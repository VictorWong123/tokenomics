// Controlled component benchmark, not language-model throughput evidence.
#include "ggml-cpu.h"
#include "ggml-cpu/quants.h"
#include "ggml.h"
#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

extern "C" void tokenomics_vec_dot_q4_pair(int, float *, float *, const void *, const void *,
                                           const void *);
using Clock = std::chrono::steady_clock;

int main(int argc, char **argv) {
    const int width = argc > 1 ? std::stoi(argv[1]) : 4096;
    const int outputs = argc > 2 ? std::stoi(argv[2]) : 4096;
    const int repeats = argc > 3 ? std::stoi(argv[3]) : 8;
    if (width < 256 || width % 256 || width > 32768 || outputs < 16 || outputs > 160000 ||
        repeats < 2 || repeats > 100)
        throw std::runtime_error("usage: bench_q4_pair [width_multiple_256 outputs repeats]");
    ggml_cpu_init();
    auto *ctx = ggml_init({1024, nullptr, true});
    if (!ctx)
        throw std::runtime_error("ggml initialization failed");
    std::mt19937 rng(76123);
    std::normal_distribution<float> normal;
    const auto wx = ggml_row_size(GGML_TYPE_Q4_K, width);
    const auto wy = ggml_row_size(GGML_TYPE_Q8_K, width);
    std::vector<uint64_t> weights((wx * outputs + 7) / 8), acts((wy * 9 + 7) / 8);
    auto *x = reinterpret_cast<char *>(weights.data());
    auto *y = reinterpret_cast<char *>(acts.data());
    std::vector<float> row(width), ref(outputs * 9), got(ref.size());
    for (int i = 0; i < outputs + 9; ++i) {
        for (auto &v : row)
            v = normal(rng);
        if (i < outputs)
            quantize_row_q4_K(row.data(), x + i * wx, width);
        else
            quantize_row_q8_K(row.data(), y + (i - outputs) * wy, width);
    }
    const auto run = [&](bool pair, int rows, std::vector<float> &out) {
        const auto start = Clock::now();
        // Match the guarded implementation's 16-weight-row cache tile.
        for (int block = 0; block < outputs; block += 16) {
            if (!pair) {
                for (int c = 0; c < rows; ++c)
                    for (int r = block; r < std::min(block + 16, outputs); ++r)
                        ggml_vec_dot_q4_K_q8_K(width, &out[c * outputs + r], 0, x + r * wx, 0,
                                               y + c * wy, 0, 1);
            } else {
                for (int r = block; r < std::min(block + 16, outputs); ++r) {
                    int c = 0;
                    for (; c + 1 < rows; c += 2)
                        tokenomics_vec_dot_q4_pair(width, &out[c * outputs + r],
                                                   &out[(c + 1) * outputs + r], x + r * wx,
                                                   y + c * wy, y + (c + 1) * wy);
                    if (c < rows)
                        ggml_vec_dot_q4_K_q8_K(width, &out[c * outputs + r], 0, x + r * wx, 0,
                                               y + c * wy, 0, 1);
                }
            }
        }
        return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    };
    std::cout << std::setprecision(12);
    for (int rows : {1, 2, 3, 5, 7, 9}) {
        run(false, rows, ref);
        run(true, rows, got);
        for (int trial = 0; trial < repeats; ++trial) {
            double baseline, paired;
            if (trial % 2 == 0) {
                baseline = run(false, rows, ref);
                paired = run(true, rows, got);
            } else {
                paired = run(true, rows, got);
                baseline = run(false, rows, ref);
            }
            double checksum = 0;
            for (int i = 0; i < rows * outputs; ++i) {
                if (std::bit_cast<uint32_t>(ref[i]) != std::bit_cast<uint32_t>(got[i]))
                    throw std::runtime_error("bitwise matrix mismatch");
                checksum += got[i];
            }
            if (checksum == 0)
                throw std::runtime_error("degenerate matrix: check CPU initialization");
            std::cout << "{\"width\":" << width << ",\"outputs\":" << outputs
                      << ",\"rows\":" << rows << ",\"trial\":" << trial
                      << ",\"baseline_ms\":" << baseline << ",\"pair_ms\":" << paired
                      << ",\"checksum\":" << checksum << ",\"bitwise_equal\":true}\n";
        }
    }
    ggml_free(ctx);
}
