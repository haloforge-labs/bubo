add_library(gufo_diffusion_gemma STATIC config.cpp tokenizer.cpp chat.cpp
  image_chat.cpp
  cache_state.cpp image_prompt.cpp denoise.cpp read.cpp)
target_include_directories(gufo_diffusion_gemma PUBLIC ${PROJECT_SOURCE_DIR})
target_compile_features(gufo_diffusion_gemma PUBLIC cxx_std_20)

if(ENGINE_ENABLE_HIP)
  target_sources(gufo_diffusion_gemma PRIVATE
    kernels.hip attention.hip moe.hip soft_embed.hip dense.hip profiler.cpp
    upload.cpp weights.cpp weights_gguf.cpp weights_fp8.cpp weights_nvfp4.cpp
    engine.cpp)
  target_link_libraries(gufo_diffusion_gemma PUBLIC gufo_core hip::host)
  # Explicit FP32 arithmetic and BF16 operation boundaries.
  target_compile_options(gufo_diffusion_gemma PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3;-ffp-contract=off>)

  add_executable(gufo-diffusion-gemma
    ${PROJECT_SOURCE_DIR}/tools/models/diffusion_gemma/main.cpp)
  target_link_libraries(gufo-diffusion-gemma PRIVATE gufo_diffusion_gemma)
endif()

function(gufo_diffusion_gemma_register_tests)
  add_executable(diffusion_gemma_image_test
    ${PROJECT_SOURCE_DIR}/tests/models/diffusion_gemma/image_test.cpp)
  target_link_libraries(diffusion_gemma_image_test PRIVATE
    gufo_diffusion_gemma gufo_gemma4)
  add_test(NAME diffusion_gemma_image_test COMMAND diffusion_gemma_image_test)
  set_tests_properties(diffusion_gemma_image_test PROPERTIES
    LABELS "cpu;models;diffusion-gemma;vision")
  add_executable(diffusion_gemma_chat_test
    ${PROJECT_SOURCE_DIR}/tests/models/diffusion_gemma/chat_test.cpp)
  target_link_libraries(diffusion_gemma_chat_test PRIVATE gufo_diffusion_gemma)
  add_test(NAME diffusion_gemma_chat_test COMMAND diffusion_gemma_chat_test)
  set_tests_properties(diffusion_gemma_chat_test PROPERTIES
    LABELS "cpu;models;diffusion-gemma")
  add_executable(diffusion_gemma_tokenizer_test
    ${PROJECT_SOURCE_DIR}/tests/models/diffusion_gemma/tokenizer_test.cpp)
  target_link_libraries(diffusion_gemma_tokenizer_test PRIVATE
    gufo_diffusion_gemma)
  add_test(NAME diffusion_gemma_tokenizer_test
    COMMAND diffusion_gemma_tokenizer_test)
  set_tests_properties(diffusion_gemma_tokenizer_test PROPERTIES
    LABELS "cpu;models;diffusion-gemma;external-model" SKIP_RETURN_CODE 77)
  foreach(test cache_state denoise read)
    add_executable(diffusion_gemma_${test}_test
      ${PROJECT_SOURCE_DIR}/tests/models/diffusion_gemma/${test}_test.cpp)
    target_link_libraries(diffusion_gemma_${test}_test PRIVATE
      gufo_diffusion_gemma)
    add_test(NAME diffusion_gemma_${test}_test
      COMMAND diffusion_gemma_${test}_test)
    set_tests_properties(diffusion_gemma_${test}_test PROPERTIES
      LABELS "cpu;models;diffusion-gemma")
  endforeach()
  if(ENGINE_ENABLE_HIP)
    add_executable(diffusion_gemma_kernels_hip_test
      ${PROJECT_SOURCE_DIR}/tests/models/diffusion_gemma/kernels_hip_test.cpp)
    target_link_libraries(diffusion_gemma_kernels_hip_test PRIVATE
      gufo_diffusion_gemma)
    add_test(NAME diffusion_gemma_kernels_hip_test
      COMMAND diffusion_gemma_kernels_hip_test)
    set_tests_properties(diffusion_gemma_kernels_hip_test PROPERTIES
      LABELS "hip;gpu;gfx1151;models;diffusion-gemma" SKIP_RETURN_CODE 77)
  endif()
endfunction()
