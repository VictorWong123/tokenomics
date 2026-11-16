// Original pinned Q8_0 arithmetic, repeated inside one threadgroup. The group
// condition is uniform: only constant arguments and threadgroup coordinates
// participate. Synchronize before reusing the original reduction scratch.
static_assert(TOKENOMICS_METAL_Q8_GROUP_ROWS == 2, "Review named kernel group size");
[[host_name("kernel_mul_mv_q8_0_f32_grouped2")]]
kernel void kernel_mul_mv_q8_0_f32_grouped2(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint first = TOKENOMICS_METAL_Q8_GROUP_ROWS*tgpig.y;
    FOR_UNROLL (uint offset = 0; offset < TOKENOMICS_METAL_Q8_GROUP_ROWS; ++offset) {
        if (first + offset < uint(args.ne11)) {
            if (offset > 0) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            kernel_mul_mv_q8_0_f32_impl<N_R0_Q8_0, constant ggml_metal_kargs_mul_mv &>(
                args, src0, src1, dst, shmem,
                uint3(tgpig.x, first + offset, tgpig.z), tiisg, sgitg);
        }
    }
}
