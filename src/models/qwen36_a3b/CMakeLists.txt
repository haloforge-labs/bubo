# Qwen3.6-35B-A3B (GGUF arch qwen35moe): its own runtime, isolated from
# Flash-Next. It links Flash-Next's kernel libraries read-only; no Flash-Next
# source is compiled differently. gufo.exe links the library (text_runner:
# `gufo serve` of qwen35moe models) when GUFO_WITH_A3B is on (root
# CMakeLists); gufo-a3b is the bench and diagnostics driver.
if (NOT ENGINE_ENABLE_HIP)
  return()
endif()

add_library(gufo_qwen36_a3b STATIC EXCLUDE_FROM_ALL
  engine.cpp
  kernels.hip
  prefill.cpp
  checkpoint.cpp
  state.cpp
  decode_kernels.hip
  attention.hip
  prefill_kernels.hip
  dflash.cpp
  dflash_kernels.hip
  text_runner.cpp)
set_source_files_properties(kernels.hip decode_kernels.hip attention.hip
                            prefill_kernels.hip dflash_kernels.hip
                            PROPERTIES LANGUAGE HIP)
target_compile_options(gufo_qwen36_a3b PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O2>
  $<$<COMPILE_LANGUAGE:HIP>:-O3;-Wno-unused-command-line-argument>)
target_link_libraries(gufo_qwen36_a3b PUBLIC
  gufo_qwen38_flash_next gufo_qwen38_flash_next_mmq gufo_core
  hip::host roc::hipblas)

add_executable(gufo-a3b EXCLUDE_FROM_ALL main.cpp)
# Next to gufo.exe, where build.ps1 stages the ROCm and vcpkg DLLs.
set_target_properties(gufo-a3b PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
target_compile_options(gufo-a3b PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O2>)
target_link_libraries(gufo-a3b PRIVATE gufo_qwen36_a3b)

if(BUILD_TESTING)
  # These operator checks link the existing HIP runtime. The Get check uses
  # synthetic host regions and never calls HIP; the op check needs a GPU slot.
  foreach(name IN ITEMS weight_types kquant_ops)
    add_executable(qwen36_a3b_${name}_test EXCLUDE_FROM_ALL
      ${PROJECT_SOURCE_DIR}/tests/models/qwen36_a3b/${name}_test.cpp)
    target_link_libraries(qwen36_a3b_${name}_test PRIVATE gufo_qwen36_a3b)
    add_test(NAME qwen36_a3b.${name} COMMAND qwen36_a3b_${name}_test)
  endforeach()
  set_tests_properties(qwen36_a3b.weight_types PROPERTIES
    LABELS "qwen36_a3b;cpu")
  set_tests_properties(qwen36_a3b.kquant_ops PROPERTIES
    LABELS "qwen36_a3b;hip;gpu;gfx1151")

  # Engine session tests (state round trip, restore over a different history,
  # prefill chunk independence, greedy MTP equivalence). They load a model:
  # `a3b_session_test <model.gguf> [--quick]`. The ctest entry passes no path,
  # so it prints SKIP and exits 0; the GPU labels keep it out of `ctest -L cpu`.
  add_executable(a3b_session_test EXCLUDE_FROM_ALL
    ${PROJECT_SOURCE_DIR}/tests/models/qwen36_a3b/session_test.cpp)
  target_link_libraries(a3b_session_test PRIVATE gufo_qwen36_a3b)
  add_test(NAME qwen36_a3b.session COMMAND a3b_session_test)
  set_tests_properties(qwen36_a3b.session PROPERTIES
    LABELS "qwen36_a3b;hip;gpu;gfx1151")
endif()
