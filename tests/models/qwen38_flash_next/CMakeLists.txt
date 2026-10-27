# Small operator checks are the default. Full-model qualification and
# diagnostic probes are built explicitly so routine iterations stay focused.
add_custom_target(qwen38_flash_next_tests)

function(add_qwen38_flash_next_test name library labels)
  set(target qwen38_flash_next_${name}_test)
  add_executable(${target} ${name}_test.cpp)
  target_link_libraries(${target} PRIVATE ${library})
  add_test(NAME qwen38_flash_next.${name} COMMAND ${target})
  set_tests_properties(qwen38_flash_next.${name} PROPERTIES
    LABELS "qwen38_flash_next;${labels}")
  add_dependencies(qwen38_flash_next_tests ${target})
endfunction()

add_qwen38_flash_next_test(batch_draft_head gufo_core "cpu")
add_qwen38_flash_next_test(config gufo_qwen38_flash_next "cpu")
add_qwen38_flash_next_test(ngram gufo_qwen38_flash_next "cpu")
add_qwen38_flash_next_test(mtp_sampling gufo_core "cpu")
add_qwen38_flash_next_test(batch_twin_policy gufo_core "cpu")
add_qwen38_flash_next_test(iq4_wide_policy gufo_core "cpu")
add_qwen38_flash_next_test(batch_twin_refusal gufo_core "cpu")
add_qwen38_flash_next_test(prompt_lookup gufo_core "cpu")
add_qwen38_flash_next_test(twin_weights gufo_qwen38_flash_next "cpu")
add_qwen38_flash_next_test(c4_contracts gufo_qwen38_flash_next "cpu")
add_qwen38_flash_next_test(iq_route_policy gufo_core "cpu")

# Tokenization and manifest preparation are CPU-buildable; actual forced
# serving-batch execution is enabled only with the engine's HIP backend.
add_executable(qwen38_flash_next_batch_quality_probe EXCLUDE_FROM_ALL
  batch_quality_probe.cpp)
target_link_libraries(qwen38_flash_next_batch_quality_probe PRIVATE
  gufo_qwen38_flash_next)
if(ENGINE_ENABLE_HIP)
  target_compile_definitions(qwen38_flash_next_batch_quality_probe PRIVATE
    GUFO_BATCH_QUALITY_HIP=1)
endif()

add_executable(qwen38_flash_next_reference_probe EXCLUDE_FROM_ALL reference_probe.cpp)
target_link_libraries(qwen38_flash_next_reference_probe PRIVATE
  gufo_qwen38_flash_next_reference)
set_source_files_properties(reference_probe.cpp PROPERTIES COMPILE_OPTIONS "-O2")

if (ENGINE_ENABLE_HIP)
  foreach(name IN ITEMS mtp_ops blaslt_ops hc_mix_ops gdn_ops
      attention_ops moe_epilogue_ops moe_ids_ops projection_ops routed_wmma_ops
      iq_experts_ops
      select_ops)
    add_qwen38_flash_next_test(${name} gufo_qwen38_flash_next "hip;gpu;gfx1151")
  endforeach()

  add_executable(qwen38_flash_next_session_test EXCLUDE_FROM_ALL session_test.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/serve/continuation_cache.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/serve/continuation_disk_store.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/serve/text_model_runner.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/serve/text_generation_scheduler.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/serve/inference_backend.cpp)
  # inference_backend.cpp also serves Gemma 4 since the Gemma merge
  target_link_libraries(qwen38_flash_next_session_test PRIVATE gufo_logging
    gufo_qwen38_flash_next gufo_deepseek_v4_flash gufo_gemma4 hip::host)
  add_executable(qwen38_flash_next_snapshot_test EXCLUDE_FROM_ALL snapshot_test.cpp
    device_headroom_fault.cpp)
  target_link_libraries(qwen38_flash_next_snapshot_test PRIVATE
    gufo_qwen38_flash_next)
  # GPU retry of a faulted NPU chunk (L-REL-2): restore, chunk grid, backup
  # cost. Needs the model (GUFO_QFN_MODEL or --model), no NPU; exit 77 = skipped.
  add_executable(qwen38_flash_next_npu_retry_gpu_test EXCLUDE_FROM_ALL
    npu_retry_gpu_test.cpp)
  target_link_libraries(qwen38_flash_next_npu_retry_gpu_test PRIVATE
    gufo_qwen38_flash_next hip::host)
  add_test(NAME qwen38_flash_next.npu_retry_gpu
    COMMAND qwen38_flash_next_npu_retry_gpu_test)
  set_tests_properties(qwen38_flash_next.npu_retry_gpu PROPERTIES
    LABELS "qwen38_flash_next;hip;gpu;gfx1151;npu" SKIP_RETURN_CODE 77
    TIMEOUT 1800)
  # The executor calls the NPU prefill runtime when GUFO_NPU is defined.
  if(GUFO_ENABLE_NPU)
    foreach(npu_test IN ITEMS qwen38_flash_next_session_test
        qwen38_flash_next_snapshot_test qwen38_flash_next_npu_retry_gpu_test)
      npu_attach_runtime(${npu_test})
      if(TARGET npu_bfp_hip)
        target_link_libraries(${npu_test} PRIVATE npu_bfp_hip)
      endif()
    endforeach()
  endif()
  add_custom_target(qwen38_flash_next_model_tests
    DEPENDS qwen38_flash_next_session_test qwen38_flash_next_snapshot_test)

  add_executable(qwen38_flash_next_dense_gemm_bench EXCLUDE_FROM_ALL
    dense_gemm_bench.cpp)
  target_link_libraries(qwen38_flash_next_dense_gemm_bench PRIVATE
    gufo_qwen38_flash_next hip::host)

  add_executable(qwen38_flash_next_dense_twin_gpu_test EXCLUDE_FROM_ALL dense_twin_gpu_test.cpp)
  target_link_libraries(qwen38_flash_next_dense_twin_gpu_test PRIVATE
    gufo_qwen38_flash_next hip::host)

  add_executable(qwen38_flash_next_expert_decode_bench EXCLUDE_FROM_ALL expert_decode_bench.cpp)
  target_link_libraries(qwen38_flash_next_expert_decode_bench PRIVATE
    gufo_qwen38_flash_next hip::host)

  add_executable(qwen38_flash_next_routed_wmma_iq_bench EXCLUDE_FROM_ALL routed_wmma_iq_bench.cpp)
  target_link_libraries(qwen38_flash_next_routed_wmma_iq_bench PRIVATE
    gufo_qwen38_flash_next hip::host)

  add_executable(qwen38_flash_next_attn_bench EXCLUDE_FROM_ALL attn_bench.cpp)
  target_link_libraries(qwen38_flash_next_attn_bench PRIVATE
    gufo_qwen38_flash_next hip::host)

  add_executable(qwen38_flash_next_gpu_probe EXCLUDE_FROM_ALL gpu_probe.cpp
    mtp_audit.cpp)
  target_link_libraries(qwen38_flash_next_gpu_probe PRIVATE
    gufo_qwen38_flash_next_reference)
endif()
