# Reading a source to generate another source must also track its changes.
# Otherwise a normal incremental build can silently keep an old embedded kernel.
include_guard(GLOBAL)
macro(tokenomics_read_source path output)
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${path}")
  file(READ "${path}" ${output})
endmacro()
