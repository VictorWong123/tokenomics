cmake_minimum_required(VERSION 3.24)
project(tokenomics_metal_source_only LANGUAGES CXX)
# Generate the exact owned shader input without compiling/linking a backend.
get_filename_component(root "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
execute_process(COMMAND git rev-parse HEAD WORKING_DIRECTORY "${root}/third_party/llama.cpp"
  OUTPUT_VARIABLE pin OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
if(NOT pin STREQUAL "fd27b2034905f8dadd8bd6d2cc44595f4929d391")
  message(FATAL_ERROR "Unexpected upstream pin")
endif()
execute_process(COMMAND git status --porcelain --untracked-files=normal
  WORKING_DIRECTORY "${root}/third_party/llama.cpp"
  OUTPUT_VARIABLE dirty OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
if(NOT dirty STREQUAL "")
  message(FATAL_ERROR "Upstream submodule must be clean")
endif()
set(metal "${root}/third_party/llama.cpp/ggml/src/ggml-metal")
add_library(ggml-metal OBJECT EXCLUDE_FROM_ALL
  "${metal}/ggml-metal-common.cpp" "${metal}/ggml-metal-ops.cpp" "${metal}/ggml-metal-device.cpp")
include("${root}/cmake/metal-short-mv.cmake")
tokenomics_metal_short_mv("${root}")
set(TOKENOMICS_METAL_SHORT_MV ON)
set(GGML_METAL_EMBED_LIBRARY ON)
set(original_asm "${CMAKE_CURRENT_BINARY_DIR}/ggml-metal-embed-mul_mv.s")
file(WRITE "${original_asm}" "")
target_sources(ggml-metal PRIVATE "${original_asm}")
include("${root}/cmake/metal-interleave.cmake")
tokenomics_metal_interleave("${root}")
set_source_files_properties("${CMAKE_CURRENT_BINARY_DIR}/tokenomics-metal-embed-mul_mv.s"
  PROPERTIES HEADER_FILE_ONLY TRUE)
