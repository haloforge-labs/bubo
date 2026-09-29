# Qwen3.6-35B-A3B (GGUF arch qwen35moe): an exploration runtime, isolated
# from Flash-Next. It links Flash-Next's kernel libraries read-only and builds
# its own library and executable; no Flash-Next source is compiled
# differently. gufo.exe links the library for `gufo serve` of qwen35moe
# models when GUFO_WITH_A3B is on (root CMakeLists).
if (NOT ENGINE_ENABLE_HIP)
  return()
endif()

add_library(gufo_qwen36_a3b STATIC EXCLUDE_FROM_ALL
  engine.cpp
  kernels.hip
  prefill.cpp
  checkpoint.cpp
  decode_kernels.hip
  attention.hip
  prefill_kernels.hip
  dflash.cpp
  dflash_kernels.hip
  serve.cpp
  serve_cli.cpp)
set_source_files_properties(kernels.hip decode_kernels.hip attention.hip
                            prefill_kernels.hip dflash_kernels.hip
                            PROPERTIES LANGUAGE HIP)
target_compile_options(gufo_qwen36_a3b PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O2>
  $<$<COMPILE_LANGUAGE:HIP>:-O3;-Wno-unused-command-line-argument>)
target_link_libraries(gufo_qwen36_a3b PUBLIC
  gufo_qwen38_flash_next gufo_qwen38_flash_next_mmq gufo_http gufo_core
  hip::host roc::hipblas)

add_executable(gufo-a3b EXCLUDE_FROM_ALL main.cpp)
# Next to gufo.exe, where build.ps1 stages the ROCm and vcpkg DLLs.
set_target_properties(gufo-a3b PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
target_compile_options(gufo-a3b PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O2>)
target_link_libraries(gufo-a3b PRIVATE gufo_qwen36_a3b)
