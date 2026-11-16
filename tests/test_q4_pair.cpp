// Bitwise comparison with the pinned single-row AVX2 implementation.
#include "ggml-cpu.h"
#include "ggml-cpu/quants.h"
#include "ggml.h"
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

extern "C" void tokenomics_vec_dot_q4_pair(int, float *, float *, const void *, const void *,
                                           const void *);

int main() {
    // Initialize the half-precision conversion table on configurations that use it.
    ggml_cpu_init();
    auto *ctx = ggml_init({1024, nullptr, true});
    if (!ctx)
        throw std::runtime_error("ggml initialization failed");
    std::mt19937 rng(76123);
    std::normal_distribution<float> normal;
    int checks = 0;
    int nonzero = 0;
    for (int n : {256, 512, 768, 1024, 1792, 4096, 8192, 18944}) {
        std::vector<float> x(n), a(n), b(n);
        // Typed uint64 storage guarantees alignment without assuming SIMD alignment.
        std::vector<uint64_t> qx((ggml_row_size(GGML_TYPE_Q4_K, n) + 7) / 8);
        std::vector<uint64_t> qa((ggml_row_size(GGML_TYPE_Q8_K, n) + 7) / 8), qb(qa.size());
        for (int trial = 0; trial < 128; ++trial) {
            for (int i = 0; i < n; ++i) {
                x[i] = trial == 0 ? 0.0f : normal(rng) * std::pow(2.0f, trial % 13 - 6);
                a[i] = trial == 1 ? 0.0f : normal(rng);
                b[i] = trial == 2 ? 1.0f : normal(rng) * std::pow(2.0f, trial % 9 - 4);
            }
            quantize_row_q4_K(x.data(), qx.data(), n);
            quantize_row_q8_K(a.data(), qa.data(), n);
            quantize_row_q8_K(b.data(), qb.data(), n);
            float ref0, ref1, got0, got1;
            ggml_vec_dot_q4_K_q8_K(n, &ref0, 0, qx.data(), 0, qa.data(), 0, 1);
            ggml_vec_dot_q4_K_q8_K(n, &ref1, 0, qx.data(), 0, qb.data(), 0, 1);
            tokenomics_vec_dot_q4_pair(n, &got0, &got1, qx.data(), qa.data(), qb.data());
            if (!std::isfinite(got0) || !std::isfinite(got1) ||
                std::bit_cast<uint32_t>(ref0) != std::bit_cast<uint32_t>(got0) ||
                std::bit_cast<uint32_t>(ref1) != std::bit_cast<uint32_t>(got1)) {
                std::cerr << "bitwise Q4 pair mismatch n=" << n << " trial=" << trial << '\n';
                return 1;
            }
            checks += 2;
            nonzero += got0 != 0;
            nonzero += got1 != 0;
        }
    }
    ggml_free(ctx);
    if (nonzero < checks / 2)
        throw std::runtime_error("degenerate dot products: check CPU initialization");
    std::cout << checks << " bitwise Q4 pair comparisons passed\n";
}
