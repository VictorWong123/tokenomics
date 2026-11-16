// Experimental Q8_0 weight loads shared across two query rows. Based on the
// pinned MIT-licensed ggml-metal/kernels/mul_mv.metal Q8 routine. Keep each
// query's eight-value accumulation, block scale and reduction helper in order.
template<short NR0, typename args_t>
void tokenomics_mul_mv_q8_shared_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3 tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;
    constexpr short NW = N_SIMDWIDTH;
    constexpr short NQ = 8;
    const int nb = args.ne00/QK8_0;
    const int r0 = tgpig.x*NR0;
    const int r1 = 2*tgpig.y;
    const int im = tgpig.z;
    const bool second = r1 + 1 < args.ne11;
    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;
    const uint64_t offset1 = r1*args.nb11 + i12*args.nb12 + i13*args.nb13;
    device const float * y = (device const float *) (src1 + offset1);
    device const float * y_next = second ? (device const float *) (src1 + offset1 + args.nb11) : y;
    device const block_q8_0 * ax[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        const uint64_t offset0 = (r0 + row)*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
        ax[row] = (device const block_q8_0 *) (src0 + offset0);
    }
    float sumf[NR0] = { 0.f };
    float sumf_next[NR0] = { 0.f };
    const short ix = tiisg/(NW/NQ);
    const short il = tiisg%(NW/NQ);
    const int ib0 = sgitg*NQ + ix;
    device const float * yb = y + ib0*QK8_0 + il*NQ;
    device const float * yb_next = y_next + ib0*QK8_0 + il*NQ;
    for (int ib = ib0; ib < nb; ib += NSG*NQ) {
        float yl[NQ];
        float yl_next[NQ];
        for (short i = 0; i < NQ; ++i) {
            yl[i] = yb[i];
            if (second) {
                yl_next[i] = yb_next[i];
            }
        }
        for (short row = 0; row < NR0; ++row) {
            device const int8_t * qs = ax[row][ib].qs + il*NQ;
            const float scale = ax[row][ib].d;
            float sumq = 0.f;
            float sumq_next = 0.f;
            FOR_UNROLL (short i = 0; i < NQ; ++i) {
                const float q = qs[i];
                sumq += q*yl[i];
                if (second) {
                    sumq_next += q*yl_next[i];
                }
            }
            sumf[row] += sumq*scale;
            if (second) {
                sumf_next[row] += sumq_next*scale;
            }
        }
        yb += NSG*NQ*QK8_0;
        if (second) {
            yb_next += NSG*NQ*QK8_0;
        }
    }
    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;
    helper_mv_reduce_and_write<NR0>(dst_f32, sumf, r0, args.ne01, tiisg, sgitg, shmem);
    if (second) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        helper_mv_reduce_and_write<NR0>(dst_f32 + args.ne0, sumf_next, r0, args.ne01, tiisg, sgitg, shmem);
    }
}
#ifndef TOKENOMICS_METAL_HOST_ARITHMETIC
[[host_name("kernel_mul_mv_q8_0_f32_shared2")]]
kernel void kernel_mul_mv_q8_0_f32_shared2(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    tokenomics_mul_mv_q8_shared_impl<N_R0_Q8_0, constant ggml_metal_kargs_mul_mv &>(
        args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}
#endif
