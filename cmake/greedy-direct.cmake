# Build a narrow, opt-in copy of the pinned common sampler. Verification and
# acceptance stay upstream; only eligible target argmax avoids token_data setup.
include_guard(GLOBAL)
function(tokenomics_greedy_direct root)
  set(SAMPLING_DIR "${root}/third_party/llama.cpp/common")
  tokenomics_read_source("${SAMPLING_DIR}/sampling.cpp" SOURCE)
  tokenomics_read_source("${root}/src/greedy_direct_sampling.inc" DIRECT_BODY)
  set(FIELD_ANCHOR "    mutable int64_t t_total_us = 0;")
  set(API_ANCHOR "llama_token common_sampler_sample(struct common_sampler * gsmpl, struct llama_context * ctx, int idx, bool grammar_first) {")
  foreach(ANCHOR FIELD_ANCHOR API_ANCHOR)
    string(FIND "${SOURCE}" "${${ANCHOR}}" POSITION)
    if(POSITION EQUAL -1)
      message(FATAL_ERROR "Pinned common sampler ${ANCHOR} changed; review direct greedy")
    endif()
  endforeach()
  # This statement occurs twice (initial selection and grammar resampling).
  # Insert at the function's entry instead, after its existing synchronization
  # and time measurement. Ineligible samplers preserve the complete old path.
  set(TIME_ANCHOR "    const auto tm = gsmpl->tm();\n\n    llama_token id = LLAMA_TOKEN_NULL;")
  string(FIND "${SOURCE}" "${TIME_ANCHOR}" POSITION)
  if(POSITION EQUAL -1)
    message(FATAL_ERROR "Pinned sampling entry changed; review direct greedy")
  endif()
  string(REPLACE "${FIELD_ANCHOR}"
    "    bool tokenomics_direct = false;\n    uint64_t tokenomics_direct_calls = 0;\n${FIELD_ANCHOR}" SOURCE "${SOURCE}")
  string(REPLACE "${API_ANCHOR}" "${DIRECT_BODY}\n${API_ANCHOR}" SOURCE "${SOURCE}")
  set(DIRECT_SAMPLE [=[    if (gsmpl->tokenomics_direct &&
        llama_get_sampled_token_ith(ctx, idx) == LLAMA_TOKEN_NULL &&
        !llama_get_sampled_probs_ith(ctx, idx) &&
        !llama_get_sampled_logits_ith(ctx, idx)) {
        const auto * logits = llama_get_logits_ith(ctx, idx);
        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
        const auto selected = tokenomics::strict_greedy_argmax(logits, n_vocab);
        // This private adapter path exposes only the selected candidate. Its
        // caller uses greedy sample/accept, never distributions or rejection.
        gsmpl->cur.resize(1);
        gsmpl->cur[0] = {selected, logits[selected], 0.0f};
        gsmpl->cur_p = {gsmpl->cur.data(), 1, 0, false};
        ++gsmpl->tokenomics_direct_calls;
        return selected;
    }
]=])
  string(REPLACE "${TIME_ANCHOR}" "${TIME_ANCHOR}\n\n${DIRECT_SAMPLE}" SOURCE "${SOURCE}")
  # Copies and clones deliberately start with the optimization disabled. The
  # private adapter enables only its fresh target sampler, never draft copies.
  set(COPY_ANCHOR "    dst->t_total_us = src->t_total_us;")
  string(FIND "${SOURCE}" "${COPY_ANCHOR}" POSITION)
  if(POSITION EQUAL -1)
    message(FATAL_ERROR "Pinned sampler copy changed; review direct greedy")
  endif()
  string(REPLACE "${COPY_ANCHOR}" "${COPY_ANCHOR}\n    dst->tokenomics_direct = false;\n    dst->tokenomics_direct_calls = 0;" SOURCE "${SOURCE}")
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/tokenomics-sampling.cpp"
    "#include \"greedy_direct.h\"\n#include \"greedy_direct_sampling.h\"\n${SOURCE}")
  get_target_property(SOURCES llama-common SOURCES)
  list(LENGTH SOURCES ORIGINAL_COUNT)
  list(FILTER SOURCES EXCLUDE REGEX "(^|/)sampling\\.cpp$")
  list(LENGTH SOURCES GUARDED_COUNT)
  math(EXPR EXPECTED_COUNT "${ORIGINAL_COUNT} - 1")
  if(NOT GUARDED_COUNT EQUAL EXPECTED_COUNT)
    message(FATAL_ERROR "Could not isolate pinned common sampling source")
  endif()
  set_property(TARGET llama-common PROPERTY SOURCES ${SOURCES})
  target_sources(llama-common PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/tokenomics-sampling.cpp")
  target_include_directories(llama-common PRIVATE "${root}/src" "${SAMPLING_DIR}")
endfunction()
