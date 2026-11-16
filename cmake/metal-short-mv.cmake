include("${CMAKE_CURRENT_LIST_DIR}/source-input.cmake")
# Generate owned copies; do not modify the pinned upstream submodule.
function(tokenomics_metal_short_mv root)
  set(metal_dir "${root}/third_party/llama.cpp/ggml/src/ggml-metal")
  foreach(unit common ops)
    tokenomics_read_source("${metal_dir}/ggml-metal-${unit}.cpp" source)
    set(source "#include \"${root}/src/metal_short_mv.h\"\n${source}")
    if(unit STREQUAL "common")
      # Keep supports_op and runtime fusion eligibility consistent with dispatch.
      # The structural fusion predicate deliberately remains shape-independent,
      # like upstream; actual fusion already consults use_mma for each graph.
      set(anchors
        "bool ggml_metal_op_mul_mat_use_mm(const struct ggml_tensor * op, bool has_simdgroup_mm) {"
        "bool ggml_metal_op_mul_mat_use_mma(const struct ggml_tensor * op) {")
      foreach(anchor IN LISTS anchors)
        string(FIND "${source}" "${anchor}" position)
        if(position EQUAL -1)
          message(FATAL_ERROR "Pinned Metal selector changed: ${anchor}")
        endif()
        string(REPLACE "${anchor}" "${anchor}\n    if (tokenomics_metal_short_mv(op)) { return false; }" source "${source}")
      endforeach()
    else()
      set(anchor "    if (props_dev->has_simdgroup_mm && ggml_metal_op_mul_mat_use_mma(op)) {")
      string(FIND "${source}" "${anchor}" position)
      if(position EQUAL -1)
        message(FATAL_ERROR "Pinned Metal matmul dispatch changed")
      endif()
      string(REPLACE "${anchor}"
        "    // Preserve ordinary per-row mat-vec arithmetic across short batches.\n    if (tokenomics_metal_short_mv(op)) {\n        return ggml_metal_op_mul_mat_mv(ctx, idx);\n    }\n\n${anchor}"
        source "${source}")
      # Softmax also changes its reduction geometry with the query-row count.
      # Match ordinary one-row decode for short batches, retaining head count.
      set(softmax_anchor [=[
    int nth = 32; // SIMD width

    if (ne00%4 == 0) {
        while (nth < ne00/4 && nth*ne01*ne02*ne03 < 256) {
            nth *= 2;
        }
    } else {
        while (nth < ne00 && nth*ne01*ne02*ne03 < 256) {
            nth *= 2;
        }
    }
]=])
      string(FIND "${source}" "${softmax_anchor}" softmax_position)
      if(softmax_position EQUAL -1)
        message(FATAL_ERROR "Pinned Metal softmax thread selection changed")
      endif()
      string(REPLACE "nth*ne01*ne02*ne03 < 256"
        "nth*(tokenomics::metal_verification_graph_op(op) && ne01 <= 9 ? 1 : ne01)*ne02*ne03 < 256" softmax_replacement "${softmax_anchor}")
      string(REPLACE "${softmax_anchor}" "${softmax_replacement}" source "${source}")
    endif()
    set(generated "${CMAKE_CURRENT_BINARY_DIR}/tokenomics-metal-${unit}.cpp")
    file(WRITE "${generated}" "${source}")
    get_target_property(sources ggml-metal SOURCES)
    list(LENGTH sources before)
    list(FILTER sources EXCLUDE REGEX "(^|/)ggml-metal-${unit}\\.cpp$")
    list(LENGTH sources after)
    math(EXPR expected "${before} - 1")
    if(NOT after EQUAL expected)
      message(FATAL_ERROR "Could not isolate pinned Metal ${unit} source")
    endif()
    set_property(TARGET ggml-metal PROPERTY SOURCES ${sources})
    target_sources(ggml-metal PRIVATE "${generated}")
  endforeach()
  target_include_directories(ggml-metal PRIVATE "${metal_dir}")
endfunction()
