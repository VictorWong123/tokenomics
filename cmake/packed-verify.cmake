# Owned integration against the checked, unmodified llama.cpp pin. Keep native
# packed matrix-vector arithmetic for every verification row, including its
# activation quantizer. Large prefills retain upstream's matrix path.
# The pinned dispatcher chunks src1 in complete planes: nrows == ne11.
# Keep its quantization-layout and GEMV/GEMM thresholds synchronized.
function(tokenomics_packed_verify)
  set(cpu_dir "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/ggml/src/ggml-cpu")
  file(READ "${cpu_dir}/repack.cpp" source)
  set(old1 "if (nrows > 3) {")
  set(new1 "if (nrows > 9) {")
  set(old2 "for (int iter = nrows - (nrows % 4); iter < nrows; iter++) {")
  set(new2 "for (int iter = nrows > 9 ? nrows - (nrows % 4) : 0; iter < nrows; iter++) {")
  set(old3 "for (int64_t i11 = ith * 4; i11 < ne11 - ne11 % 4; i11 += nth * 4) {")
  set(new3 "for (int64_t i11 = ith * 4; i11 < (ne11 > 9 ? ne11 - ne11 % 4 : 0); i11 += nth * 4) {")
  set(old4 "const int64_t i11_processed = ne11 - ne11 % 4;")
  set(new4 "const int64_t i11_processed = ne11 > 9 ? ne11 - ne11 % 4 : 0;")
  foreach(index RANGE 1 4)
    string(FIND "${source}" "${old${index}}" position)
    if(position EQUAL -1)
      message(FATAL_ERROR "Pinned repack integration point ${index} changed")
    endif()
    string(REPLACE "${old${index}}" "${new${index}}" source "${source}")
  endforeach()
  string(REPLACE "// If there are more than three rows in src1, use gemm; otherwise, use gemv."
    "// Tokenomics: use identical row quantization and GEMV through nine verification rows."
    source "${source}")
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/tokenomics-repack.cpp" "${source}")
  get_target_property(cpu_sources ggml-cpu SOURCES)
  list(LENGTH cpu_sources before)
  list(FILTER cpu_sources EXCLUDE REGEX "(^|/)ggml-cpu/repack\\.cpp$")
  list(LENGTH cpu_sources after)
  math(EXPR expected "${before} - 1")
  if(NOT after EQUAL expected)
    message(FATAL_ERROR "Could not isolate the pinned CPU repack source")
  endif()
  set_property(TARGET ggml-cpu PROPERTY SOURCES ${cpu_sources})
  target_sources(ggml-cpu PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/tokenomics-repack.cpp")
  target_include_directories(ggml-cpu PRIVATE "${cpu_dir}")
endfunction()
