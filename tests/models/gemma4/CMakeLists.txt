# Host contracts run in the hosted PR suite. Model-backed checks read the
# GGUF named by GUFO_GEMMA4_MODEL and skip (77) without it.
add_custom_target(gemma4_tests)

function(add_gemma4_test name library labels)
  set(target gemma4_${name}_test)
  add_executable(${target} ${name}_test.cpp)
  target_link_libraries(${target} PRIVATE ${library})
  add_test(NAME gemma4.${name} COMMAND ${target})
  set_tests_properties(gemma4.${name} PROPERTIES
    LABELS "gemma4;${labels}" SKIP_RETURN_CODE 77)
  add_dependencies(gemma4_tests ${target})
endfunction()

add_gemma4_test(config gufo_gemma4 "cpu")
add_gemma4_test(draft_policy gufo_gemma4 "cpu;speculative")
add_gemma4_test(draft_upload gufo_gemma4 "cpu;speculative")
add_gemma4_test(draft_vocab_policy gufo_gemma4 "cpu;speculative")
add_test(NAME gemma4.draft_vocab_default_guard
  COMMAND gemma4_draft_vocab_policy_test --default-guard)
set_tests_properties(gemma4.draft_vocab_default_guard PROPERTIES
  LABELS "gemma4;cpu;speculative")
add_gemma4_test(draft_vocab_quant gufo_gemma4 "cpu;speculative")
if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  # Host binary16 casts use Clang's matching runtime; the Windows C runtime
  # does not supply __truncsfhf2 / __extendhfsf2.
  execute_process(COMMAND "${CMAKE_CXX_COMPILER}" --rtlib=compiler-rt --print-libgcc-file-name
    OUTPUT_VARIABLE gemma4_host_builtins OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
  foreach(target gemma4_draft_vocab_policy_test gemma4_draft_vocab_quant_test)
    target_link_libraries(${target} PRIVATE "${gemma4_host_builtins}")
  endforeach()
endif()
add_test(NAME gemma4.draft_vocab_quant_range
  COMMAND gemma4_draft_vocab_quant_test --q4-k-range)
set_tests_properties(gemma4.draft_vocab_quant_range PROPERTIES
  LABELS "gemma4;cpu;speculative")
add_gemma4_test(tokenizer_contract gufo_gemma4 "cpu;tokenization")
add_gemma4_test(template gufo_gemma4 "cpu;chat_template")
add_gemma4_test(tool_syntax gufo_gemma4 "cpu;tools")
add_gemma4_test(vision_prompt gufo_gemma4 "cpu;vision")
add_gemma4_test(vision_weights gufo_gemma4 "cpu;vision")
add_gemma4_test(artifact gufo_gemma4 "cpu;tokenization;chat_template;external-model")
foreach(target gemma4_template_test gemma4_artifact_test)
  target_compile_definitions(${target} PRIVATE
    GUFO_CHAT_TEMPLATE_HF_GOLDENS="${PROJECT_SOURCE_DIR}/tests/fixtures/chat_template_hf_goldens.json"
    GUFO_GEMMA4_FIXTURES="${PROJECT_SOURCE_DIR}/tests/models/gemma4/fixtures")
endforeach()

add_executable(gemma4_reference_probe EXCLUDE_FROM_ALL reference_probe.cpp)
target_link_libraries(gemma4_reference_probe PRIVATE gufo_gemma4_reference)
set_source_files_properties(reference_probe.cpp PROPERTIES COMPILE_OPTIONS "-O2")
add_executable(gemma4_vision_probe EXCLUDE_FROM_ALL vision_probe.cpp)
target_link_libraries(gemma4_vision_probe PRIVATE gufo_gemma4_reference)
if (ENGINE_ENABLE_HIP)
  target_compile_definitions(gemma4_vision_probe PRIVATE GUFO_GEMMA4_VISION_GPU)
endif()

if (ENGINE_ENABLE_HIP)
  foreach(name IN ITEMS attention_ops draft_bf16_ops kernel_ops moe_ops
      projection_ops)
    add_gemma4_test(${name} gufo_gemma4 "hip;gpu;gfx1151")
  endforeach()
  set_tests_properties(gemma4.draft_bf16_ops PROPERTIES RESOURCE_LOCK gemma4)
  # The low-headroom case uses LimitReportedDeviceFree from engine.cpp (the
  # fork's dlsym(RTLD_NEXT) hipMemGetInfo interposer cannot work on Windows).
  add_executable(gemma4_target_test EXCLUDE_FROM_ALL target_test.cpp)
  target_link_libraries(gemma4_target_test PRIVATE gufo_gemma4
    gufo_gemma4_reference)
  set_source_files_properties(target_test.cpp PROPERTIES COMPILE_OPTIONS "-O2")
  target_compile_definitions(gemma4_target_test PRIVATE
    GUFO_GEMMA4_FIXTURES="${PROJECT_SOURCE_DIR}/tests/models/gemma4/fixtures")
  add_test(NAME gemma4.target COMMAND gemma4_target_test)
  set_tests_properties(gemma4.target PROPERTIES
    LABELS "gemma4;hip;gpu;gfx1151;quality;external-model;slow"
    SKIP_RETURN_CODE 77 RESOURCE_LOCK gemma4)
  add_executable(gemma4_vision_encoder_test vision_encoder_test.cpp)
  target_link_libraries(gemma4_vision_encoder_test PRIVATE
    gufo_gemma4_reference)
  set_source_files_properties(vision_encoder_test.cpp PROPERTIES
    COMPILE_OPTIONS "-O2")
  add_test(NAME gemma4.vision_encoder COMMAND gemma4_vision_encoder_test)
  set_tests_properties(gemma4.vision_encoder PROPERTIES
    LABELS "gemma4;hip;gpu;gfx1151;vision;external-model"
    SKIP_RETURN_CODE 77 RESOURCE_LOCK gemma4)
  add_executable(gemma4_vision_session_test EXCLUDE_FROM_ALL
    vision_session_test.cpp)
  target_link_libraries(gemma4_vision_session_test PRIVATE gufo_gemma4)
  add_test(NAME gemma4.vision_session COMMAND gemma4_vision_session_test)
  set_tests_properties(gemma4.vision_session PROPERTIES
    LABELS "gemma4;hip;gpu;gfx1151;vision;external-model;slow"
    SKIP_RETURN_CODE 77 RESOURCE_LOCK gemma4)
  add_executable(gemma4_vision_generate_probe EXCLUDE_FROM_ALL
    vision_generate_probe.cpp)
  target_link_libraries(gemma4_vision_generate_probe PRIVATE gufo_gemma4)
  add_custom_target(gemma4_model_tests DEPENDS gemma4_target_test
    gemma4_vision_session_test)
  add_executable(gemma4_gpu_probe EXCLUDE_FROM_ALL gpu_probe.cpp)
  target_link_libraries(gemma4_gpu_probe PRIVATE gufo_gemma4)
endif()
