// Experimental two-activation Q4_K dot product. Included in a generated copy of
// pinned ggml-cpu/arch/x86/quants.c so its SIMD helpers and types remain private.
// Adapted from that file's ggml_vec_dot_q4_K_q8_K (MIT, llama.cpp contributors).
// Each lane follows the original AVX2 integer sums, FMA and final reduction;
// only weight loads/unpacking are shared. Never used for ordinary one-row decode.
#if defined(__AVX2__)
void tokenomics_vec_dot_q4_pair(int n, float *s0, float *s1, const void *vx, const void *vy0,
                                const void *vy1);
void tokenomics_vec_dot_q4_pair(int n, float *s0, float *s1, const void *vx, const void *vy0,
                                const void *vy1) {
    assert(n % QK_K == 0);
    const block_q4_K *x = vx;
    const block_q8_K *y0 = vy0;
    const block_q8_K *y1 = vy1;
    const __m256i m4 = _mm256_set1_epi8(15);
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    __m128 min0 = _mm_setzero_ps(), min1 = _mm_setzero_ps();
    for (int i = 0; i < n / QK_K; ++i) {
        uint32_t u[4];
        memcpy(u, x[i].scales, 12);
        u[3] = ((u[2] >> 4) & 0x0f0f0f0f) | (((u[1] >> 6) & 0x03030303) << 4);
        const uint32_t aux = u[1] & 0x3f3f3f3f;
        u[1] = (u[2] & 0x0f0f0f0f) | (((u[0] >> 6) & 0x03030303) << 4);
        u[2] = aux;
        u[0] &= 0x3f3f3f3f;
        const __m256i ms = _mm256_cvtepu8_epi16(_mm_set_epi32(u[3], u[2], u[1], u[0]));
        const __m128i sc = _mm256_extracti128_si256(ms, 0);
        const __m256i scales = MM256_SET_M128I(sc, sc);
        const __m128i mins = _mm256_extracti128_si256(ms, 1);
        const float dx = GGML_CPU_FP16_TO_FP32(x[i].d);
        const float dm = GGML_CPU_FP16_TO_FP32(x[i].dmin);
        const __m256i bs0 = _mm256_loadu_si256((const __m256i *)y0[i].bsums);
        const __m256i bs1 = _mm256_loadu_si256((const __m256i *)y1[i].bsums);
        const __m128i sum0 =
            _mm_hadd_epi16(_mm256_extracti128_si256(bs0, 0), _mm256_extracti128_si256(bs0, 1));
        const __m128i sum1 =
            _mm_hadd_epi16(_mm256_extracti128_si256(bs1, 0), _mm256_extracti128_si256(bs1, 1));
        min0 = _mm_fmadd_ps(_mm_set1_ps(-y0[i].d * dm), _mm_cvtepi32_ps(_mm_madd_epi16(mins, sum0)),
                            min0);
        min1 = _mm_fmadd_ps(_mm_set1_ps(-y1[i].d * dm), _mm_cvtepi32_ps(_mm_madd_epi16(mins, sum1)),
                            min1);
        __m256i dot0 = _mm256_setzero_si256(), dot1 = _mm256_setzero_si256();
        for (int j = 0; j < QK_K / 64; ++j) {
            const __m256i sl = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j));
            const __m256i sh = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 1));
            const __m256i bits = _mm256_loadu_si256((const __m256i *)(x[i].qs + 32 * j));
            const __m256i lo = _mm256_and_si256(bits, m4);
            const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(bits, 4), m4);
            const __m256i a0 = _mm256_loadu_si256((const __m256i *)(y0[i].qs + 64 * j));
            const __m256i b0 = _mm256_loadu_si256((const __m256i *)(y0[i].qs + 64 * j + 32));
            const __m256i a1 = _mm256_loadu_si256((const __m256i *)(y1[i].qs + 64 * j));
            const __m256i b1 = _mm256_loadu_si256((const __m256i *)(y1[i].qs + 64 * j + 32));
            const __m256i p0 = _mm256_madd_epi16(sl, _mm256_maddubs_epi16(lo, a0));
            const __m256i q0 = _mm256_madd_epi16(sh, _mm256_maddubs_epi16(hi, b0));
            const __m256i p1 = _mm256_madd_epi16(sl, _mm256_maddubs_epi16(lo, a1));
            const __m256i q1 = _mm256_madd_epi16(sh, _mm256_maddubs_epi16(hi, b1));
            dot0 = _mm256_add_epi32(dot0, _mm256_add_epi32(p0, q0));
            dot1 = _mm256_add_epi32(dot1, _mm256_add_epi32(p1, q1));
        }
        acc0 = _mm256_fmadd_ps(_mm256_set1_ps(y0[i].d * dx), _mm256_cvtepi32_ps(dot0), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_set1_ps(y1[i].d * dx), _mm256_cvtepi32_ps(dot1), acc1);
    }
    min0 = _mm_add_ps(min0, _mm_movehl_ps(min0, min0));
    min0 = _mm_add_ss(min0, _mm_movehdup_ps(min0));
    min1 = _mm_add_ps(min1, _mm_movehl_ps(min1, min1));
    min1 = _mm_add_ss(min1, _mm_movehdup_ps(min1));
    *s0 = hsum_float_8(acc0) + _mm_cvtss_f32(min0);
    *s1 = hsum_float_8(acc1) + _mm_cvtss_f32(min1);
}
#endif
