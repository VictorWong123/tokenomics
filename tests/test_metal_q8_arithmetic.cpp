// Compile actual shader loop source as scalar C++. Capture partial sums before
// the unchanged Metal reduction helper; this does not emulate GPU reductions.
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

// The upstream parameter header expects fixed-width integer definitions first.
#include "ggml-metal-impl.h"

using uint = unsigned;
using ushort = unsigned short;
struct uint3 {
    uint x, y, z;
};
constexpr short N_SIMDWIDTH = 32;
constexpr short QK8_0 = 32;
static short FC_mul_mv_nsg;
constexpr short FC_mul_mv_ne12 = 1, FC_mul_mv_r2 = 1, FC_mul_mv_r3 = 1;
struct block_q8_0 {
    _Float16 d;
    int8_t qs[QK8_0];
};
static_assert(sizeof(block_q8_0) == 34);
struct Capture {
    float *destination;
    int outputs, rows, lanes;
    std::vector<float> partials;
    std::vector<uint8_t> writes;
    int barriers = 0;
    Capture(float *dst, int out, int r)
        : destination(dst), outputs(out), rows(r), lanes(FC_mul_mv_nsg * N_SIMDWIDTH),
          partials(size_t(out) * r * lanes), writes(partials.size(), 0) {}
};
static Capture *capture;
static void require(bool ok, const char *message) {
    if (!ok)
        throw std::runtime_error(message);
}
template <short NR0>
static void helper_mv_reduce_and_write(float *dst, float sumf[NR0], int r0, int outputs,
                                       ushort lane, ushort group, char *) {
    const auto delta = dst - capture->destination;
    require(delta >= 0 && delta % outputs == 0 && delta / outputs < capture->rows,
            "invalid shader output pointer");
    const int query = int(delta / outputs);
    for (int row = 0; row < NR0 && r0 + row < outputs; ++row) {
        const auto index =
            (size_t(query) * outputs + r0 + row) * capture->lanes + group * N_SIMDWIDTH + lane;
        require(capture->writes[index]++ == 0, "duplicate partial sum");
        capture->partials[index] = sumf[row];
    }
}
namespace mem_flags {
constexpr int mem_threadgroup = 1;
}
static void threadgroup_barrier(int) { ++capture->barriers; }
#define device
#define threadgroup
#define FOR_UNROLL for
#define TOKENOMICS_METAL_HOST_ARITHMETIC
#include "../src/metal_q8_shared.metal.inc"
#include "q8-reference.inc"
#undef device
#undef threadgroup
#undef FOR_UNROLL

static size_t run_case(int width, int outputs, int rows, int family, int padding) {
    std::mt19937 rng(20261116 + width + outputs + rows + family);
    ggml_metal_kargs_mul_mv args{};
    args.ne00 = width;
    args.ne01 = args.ne0 = outputs;
    args.ne02 = args.ne12 = 1;
    args.ne10 = width;
    args.ne11 = args.ne1 = rows;
    args.nb01 = width / QK8_0 * sizeof(block_q8_0) + padding;
    args.nb11 = width * sizeof(float) + padding;
    std::vector<uint64_t> packed((args.nb01 * outputs + 7) / 8);
    auto *weights = reinterpret_cast<char *>(packed.data());
    for (int row = 0; row < outputs; ++row) {
        for (int block = 0; block < width / QK8_0; ++block) {
            auto *q = std::construct_at(reinterpret_cast<block_q8_0 *>(weights + row * args.nb01 +
                                                                       block * sizeof(block_q8_0)));
            const float scales[] = {
                0.0f, -0.0f,   0.000000059604644775390625f, 0.00006103515625f, 0.015625f, -0.5f,
                1.0f, 65504.0f};
            q->d = _Float16(scales[rng() % 8]);
            for (auto &value : q->qs)
                value = int8_t(int(rng() % 256) - 128);
        }
    }
    // Spare storage keeps the original routine's non-dereferenced inactive-lane
    // pointers within the allocated C++ array, including widths below one tile.
    std::vector<float> input(args.nb11 / sizeof(float) * rows + 4096, NAN);
    for (int row = 0; row < rows; ++row)
        for (int col = 0; col < width; ++col) {
            float value = float(int(rng() % 65536) - 32768) / 32768.0f;
            if (family == 1)
                value = col % 2 ? -1e8f : 1e8f;
            if (family == 2)
                value *= 1e-25f;
            input[row * args.nb11 / sizeof(float) + col] = value;
        }
    std::vector<float> dst(size_t(outputs) * rows);
    Capture original(dst.data(), outputs, rows), shared(dst.data(), outputs, rows);
    char scratch[32 * sizeof(float) * N_R0_Q8_0];
    capture = &original;
    for (int query = 0; query < rows; ++query)
        for (int tile = 0; tile < outputs / N_R0_Q8_0; ++tile)
            for (int group = 0; group < FC_mul_mv_nsg; ++group)
                for (int lane = 0; lane < N_SIMDWIDTH; ++lane)
                    kernel_mul_mv_q8_0_f32_impl<N_R0_Q8_0, const ggml_metal_kargs_mul_mv &>(
                        args, weights, reinterpret_cast<char *>(input.data()),
                        reinterpret_cast<char *>(dst.data()), scratch,
                        uint3{uint(tile), uint(query), 0}, lane, group);
    capture = &shared;
    for (int query_pair = 0; query_pair < (rows + 1) / 2; ++query_pair)
        for (int tile = 0; tile < outputs / N_R0_Q8_0; ++tile)
            for (int group = 0; group < FC_mul_mv_nsg; ++group)
                for (int lane = 0; lane < N_SIMDWIDTH; ++lane)
                    tokenomics_mul_mv_q8_shared_impl<N_R0_Q8_0, const ggml_metal_kargs_mul_mv &>(
                        args, weights, reinterpret_cast<char *>(input.data()),
                        reinterpret_cast<char *>(dst.data()), scratch,
                        uint3{uint(tile), uint(query_pair), 0}, lane, group);
    require(shared.barriers == rows / 2 * (outputs / N_R0_Q8_0) * shared.lanes,
            "nonuniform or missing shared-scratch barrier");
    for (size_t i = 0; i < original.partials.size(); ++i) {
        require(original.writes[i] == 1 && shared.writes[i] == 1, "missing query/lane partial");
        const float a = original.partials[i], b = shared.partials[i];
        require(std::isfinite(a) && std::isfinite(b), "nonfinite source arithmetic or poison read");
        if (std::bit_cast<uint32_t>(a) != std::bit_cast<uint32_t>(b)) {
            std::cerr << "partial mismatch width=" << width << " outputs=" << outputs
                      << " rows=" << rows << " nsg=" << FC_mul_mv_nsg << " family=" << family
                      << " padding=" << padding << " index=" << i << " original=" << a
                      << " shared=" << b << '\n';
            throw std::runtime_error("Q8 per-thread arithmetic differs");
        }
    }
    return original.partials.size();
}
int main() {
    try {
        size_t compared = 0;
        int cases = 0;
        for (int width : {32, 64, 128, 256, 896, 1536, 4864})
            for (int outputs : {2, 6})
                for (int rows = 2; rows <= 9; ++rows)
                    for (int groups : {1, 2, 4})
                        for (int family = 0; family < 3; ++family)
                            for (int padding : {0, 32}) {
                                FC_mul_mv_nsg = groups;
                                compared += run_case(width, outputs, rows, family, padding);
                                ++cases;
                            }
        std::cout << "{\"state\":\"passed\",\"cases\":" << cases
                  << ",\"compared_partials\":" << compared
                  << ",\"gpu_execution\":false,\"scope\":\"Source-derived scalar per-thread Q8 "
                     "arithmetic only; Metal reduction helper replaced by a capture stub\"}\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
