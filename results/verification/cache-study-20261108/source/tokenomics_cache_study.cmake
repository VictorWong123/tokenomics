function(tokenomics_cache_study)
  if(TOKENOMICS_STUDY_CACHE_BYTES GREATER 0)
    set(path "${CMAKE_BINARY_DIR}/tokenomics-repack.cpp")
    file(READ "${path}" source)
    set(old "int64_t chunk_size0 = (nr0 + nth_scaled - 1) / nth_scaled;")
    set(new "${old}
        const bool cache_bounded_verify = ne11 > 1 && ne11 <= 9;
        if (cache_bounded_verify) {
            const int64_t row_bytes = ggml_row_size(src0->type, ne00);
            const int64_t cache_rows = MAX(NB_COLS, (${TOKENOMICS_STUDY_CACHE_BYTES} / row_bytes / NB_COLS) * NB_COLS);
            chunk_size0 = MIN(chunk_size0, cache_rows);
        }")
    string(FIND "${source}" "${old}" position)
    if(position EQUAL -1)
      message(FATAL_ERROR "Missing study chunk integration point")
    endif()
    string(REPLACE "${old}" "${new}" source "${source}")
    set(old "if (nth == 1 || ((nchunk0 < nth || disable_chunking)")
    set(new "if ((nth == 1 && !cache_bounded_verify) || ((nchunk0 < nth || disable_chunking)")
    string(FIND "${source}" "${old}" position)
    if(position EQUAL -1)
      message(FATAL_ERROR "Missing study single-thread integration point")
    endif()
    string(REPLACE "${old}" "${new}" source "${source}")
    file(WRITE "${path}" "${source}")
  endif()
  add_executable(tokenomics_cache_study "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/tokenomics_cache_study.cpp")
  target_link_libraries(tokenomics_cache_study PRIVATE tokenomics_runtime)
endfunction()
cmake_language(DEFER CALL tokenomics_cache_study)
