include("${CMAKE_CURRENT_LIST_DIR}/source-input.cmake")
# Experimental integration; the pinned source is never edited in place.
function(tokenomics_q4_pair)
  if(NOT GGML_AVX2 OR NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
    message(FATAL_ERROR "Q4 pair experiment requires x86-64 AVX2")
  endif()
  set(cpu_dir "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/ggml/src/ggml-cpu")
  tokenomics_read_source("${cpu_dir}/arch/x86/quants.c" quants)
  string(REPLACE "\"../../quants.h\"" "\"${cpu_dir}/quants.h\"" quants "${quants}")
  string(REPLACE "\"../../ggml-cpu-impl.h\"" "\"${cpu_dir}/ggml-cpu-impl.h\"" quants "${quants}")
  string(APPEND quants "\n#include \"${CMAKE_CURRENT_SOURCE_DIR}/src/cpu_q4_pair.inc\"\n")
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/tokenomics-x86-quants.c" "${quants}")
  tokenomics_read_source("${cpu_dir}/ggml-cpu.c" cpu)
  set(anchor "    // block-tiling attempt\n    const int64_t blck_0 = 16;")
  set(replacement [=[
    // Share Q4 weight unpacking across pairs of short verification rows.
    // Dense planes only; attention/broadcast tensors retain the pinned path.
    if (type == GGML_TYPE_Q4_K && ne12 == 1 && ne13 == 1 &&
        ne02 == 1 && ne03 == 1 && ne11 >= 2 && ne11 <= 9 &&
        num_rows_per_vec_dot == 1) {
        extern void tokenomics_vec_dot_q4_pair(int, float *, float *, const void *,
                                               const void *, const void *);
        const size_t stride = src1_cont || src1->type != vec_dot_type ? row_size : nb11;
        for (int64_t r = ir0_start; r < ir0_end; ++r) {
            const char *weight = (const char *)src0->data + r * nb01;
            int64_t col = ir1_start;
            for (; col + 1 < ir1_end; col += 2) {
                float *a = (float *)((char *)dst->data + col * nb1) + r;
                float *b = (float *)((char *)dst->data + (col + 1) * nb1) + r;
                tokenomics_vec_dot_q4_pair(ne00, a, b, weight,
                    (const char *)wdata + col * stride,
                    (const char *)wdata + (col + 1) * stride);
            }
            if (col < ir1_end) {
                float *a = (float *)((char *)dst->data + col * nb1) + r;
                vec_dot(ne00, a, 0, weight, 0, (const char *)wdata + col * stride, 0, 1);
            }
        }
        return;
    }
    // block-tiling attempt
    const int64_t blck_0 = 16;
]=])
  string(FIND "${cpu}" "${anchor}" position)
  if(position EQUAL -1)
    message(FATAL_ERROR "Pinned Q4 pair integration point changed")
  endif()
  string(REPLACE "${anchor}" "${replacement}" cpu "${cpu}")
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/tokenomics-ggml-cpu.c" "${cpu}")
  get_target_property(cpu_sources ggml-cpu SOURCES)
  list(LENGTH cpu_sources before)
  list(FILTER cpu_sources EXCLUDE REGEX "(^|/)ggml-cpu/(ggml-cpu\\.c|arch/x86/quants\\.c)$")
  list(LENGTH cpu_sources after)
  math(EXPR expected "${before} - 2")
  if(NOT after EQUAL expected)
    message(FATAL_ERROR "Could not isolate pinned CPU and x86 quantization sources")
  endif()
  set_property(TARGET ggml-cpu PROPERTY SOURCES ${cpu_sources})
  target_sources(ggml-cpu PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}/tokenomics-ggml-cpu.c"
    "${CMAKE_CURRENT_BINARY_DIR}/tokenomics-x86-quants.c")
  target_include_directories(ggml-cpu PRIVATE "${cpu_dir}" "${cpu_dir}/arch/x86")
endfunction()
