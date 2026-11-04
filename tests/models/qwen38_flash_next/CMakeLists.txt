# Small operator checks are the default. Full-model qualification and
# diagnostic probes are built explicitly so routine iterations stay focused.
add_custom_target(qwen38_flash_next_tests)

add_test(NAME qwen38_flash_next.decode_tensor_budget
  COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_SOURCE_DIR}/decode_tensor_budget_test.py)
set_tests_properties(qwen38_flash_next.decode_tensor_budget PROPERTIES
  LABELS "qwen38_flash_next;cpu" TIMEOUT 30)

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
add_qwen38_flash_next_test(head_two gufo_core "cpu")
add_qwen38_flash_next_test(refusal_direction gufo_core "cpu")
target_compile_options(qwen38_flash_next_refusal_direction_test PRIVATE
  -fno-fast-math -ffp-contract=off)
add_qwen38_flash_next_test(config gufo_qwen38_flash_next "cpu")
add_qwen38_flash_next_test(coder256 gufo_qwen38_flash_next_reference "cpu")
add_qwen38_flash_next_test(mmq_scratch gufo_core "cpu;memory")
add_qwen38_flash_next_test(continuation_layout gufo_qwen38_flash_next "cpu")
add_qwen38_flash_next_test(kv_cache gufo_qwen38_flash_next "cpu")
add_qwen38_flash_next_test(kv_rotation gufo_qwen38_flash_next "cpu")
target_compile_options(qwen38_flash_next_kv_rotation_test PRIVATE
  -fno-fast-math -ffp-contract=off -frounding-math)
add_qwen38_flash_next_test(rope_scaling gufo_qwen38_flash_next "cpu")
add_qwen38_flash_next_test(rope_scaling_oracle gufo_qwen38_flash_next_reference "cpu")
add_qwen38_flash_next_test(draft_vocab gufo_core "cpu")
add_qwen38_flash_next_test(ngram gufo_qwen38_flash_next "cpu")
add_qwen38_flash_next_test(decode_cycle gufo_qwen38_flash_next "cpu")
add_qwen38_flash_next_test(device_accept gufo_core "cpu")
add_qwen38_flash_next_test(draft_chain gufo_core "cpu")
add_qwen38_flash_next_test(moe_fusion gufo_core "cpu")
add_test(NAME qwen38_flash_next.ple_real_rows
  COMMAND qwen38_flash_next_decode_cycle_test --real)
set_tests_properties(qwen38_flash_next.ple_real_rows PROPERTIES
  LABELS "cpu;qwen38_flash_next" TIMEOUT 120 SKIP_RETURN_CODE 77)
add_qwen38_flash_next_test(mtp_sampling gufo_core "cpu")
add_qwen38_flash_next_test(batch_twin_policy gufo_core "cpu")
add_qwen38_flash_next_test(iq4_wide_policy gufo_core "cpu")
add_qwen38_flash_next_test(iq_decode gufo_core "cpu")
add_qwen38_flash_next_test(iq_pair_decode gufo_core "cpu")
target_compile_options(qwen38_flash_next_iq_pair_decode_test PRIVATE
  -fno-fast-math -ffp-contract=off)
add_qwen38_flash_next_test(dense_gemv gufo_core "cpu")
set_source_files_properties(dense_gemv_test.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
add_qwen38_flash_next_test(expert_bench_policy gufo_core "cpu")
add_qwen38_flash_next_test(hc_bf16_dense_policy gufo_core "cpu")
add_qwen38_flash_next_test(hc_load_q8 gufo_core "cpu")
# Host-side stress (synthetic always; real HC tensors with concurrent unbuffered readers when the files exist).
add_qwen38_flash_next_test(hc_load_q8_stress gufo_core "cpu")
set(BUBO_HCQ8_MTP_FILE "" CACHE FILEPATH
  "Q8_0 MTP GGUF for the real-model hc_load_q8_stress tests (empty: those tests are not registered)")
set(BUBO_GSQ_RCO_DIR "" CACHE PATH
  "Directory that holds the qwen38-flash-next-gsq-rco-<pack> model directories (empty: the real-model tests are not registered)")
set(HCQ8_MTP_FILE "${BUBO_HCQ8_MTP_FILE}")
foreach(pack IN ITEMS "iq3s/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S" "iq2xs/IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS")
  set(shard1 "${BUBO_GSQ_RCO_DIR}/qwen38-flash-next-gsq-rco-${pack}-00001-of-00002.gguf")
  set(shard2 "${BUBO_GSQ_RCO_DIR}/qwen38-flash-next-gsq-rco-${pack}-00002-of-00002.gguf")
  if(BUBO_GSQ_RCO_DIR AND HCQ8_MTP_FILE AND EXISTS "${shard1}" AND EXISTS "${shard2}" AND EXISTS "${HCQ8_MTP_FILE}")
    string(REGEX REPLACE "/.*" "" pack_name "${pack}")
    add_test(NAME qwen38_flash_next.hc_load_q8_stress_${pack_name}
      COMMAND qwen38_flash_next_hc_load_q8_stress_test --reps 4 --real "${shard1}" "${shard2}" --mtp "${HCQ8_MTP_FILE}")
    set_tests_properties(qwen38_flash_next.hc_load_q8_stress_${pack_name} PROPERTIES
      LABELS "qwen38_flash_next;cpu;real-model" TIMEOUT 900)
  endif()
endforeach()
# Crops of the real BF16 HC tensors, when the native files are on this machine.
foreach(pack IN ITEMS "iq3s/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S" "iq2xs/IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS")
  set(shard1 "${BUBO_GSQ_RCO_DIR}/qwen38-flash-next-gsq-rco-${pack}-00001-of-00002.gguf")
  set(shard2 "${BUBO_GSQ_RCO_DIR}/qwen38-flash-next-gsq-rco-${pack}-00002-of-00002.gguf")
  if(BUBO_GSQ_RCO_DIR AND EXISTS "${shard1}" AND EXISTS "${shard2}")
    string(REGEX REPLACE "/.*" "" pack_name "${pack}")
    add_test(NAME qwen38_flash_next.hc_load_q8_real_${pack_name}
      COMMAND qwen38_flash_next_hc_load_q8_test --real "${shard1}" "${shard2}")
    set_tests_properties(qwen38_flash_next.hc_load_q8_real_${pack_name} PROPERTIES
      LABELS "qwen38_flash_next;cpu;real-model" TIMEOUT 300)
  endif()
endforeach()
add_qwen38_flash_next_test(batch_twin_refusal gufo_core "cpu")
add_qwen38_flash_next_test(prompt_lookup gufo_core "cpu")
add_qwen38_flash_next_test(twin_weights gufo_qwen38_flash_next "cpu")
add_qwen38_flash_next_test(c4_contracts gufo_qwen38_flash_next "cpu")
add_qwen38_flash_next_test(iq_route_policy gufo_core "cpu")
add_qwen38_flash_next_test(prefill_checkpoint_window gufo_core "cpu")
add_qwen38_flash_next_test(prefill_request_plan gufo_core "cpu")
add_qwen38_flash_next_test(prefill_experts gufo_qwen38_flash_next "cpu")
add_qwen38_flash_next_test(prefill_telemetry gufo_core "cpu")
add_qwen38_flash_next_test(prefill_stageb gufo_core "cpu")
add_qwen38_flash_next_test(prefill_diag_routed_type gufo_core "cpu")
add_qwen38_flash_next_test(scratch_bytes gufo_core "cpu;memory")
add_qwen38_flash_next_test(native_dispatch gufo_core "cpu")

# One small real-weight CPU comparison; no generated model fixtures or device.
add_executable(qwen38_flash_next_native_cpu_test native_cpu_test.cpp
  ${PROJECT_SOURCE_DIR}/src/models/qwen38_flash_next/config.cpp
  ${PROJECT_SOURCE_DIR}/src/models/qwen38_flash_next/weights.cpp
  ${PROJECT_SOURCE_DIR}/src/models/qwen38_flash_next/cpu_ops.cpp
  ${PROJECT_SOURCE_DIR}/src/models/qwen38_flash_next/rope_scaling.cpp)
target_link_libraries(qwen38_flash_next_native_cpu_test PRIVATE gufo_core)
add_test(NAME qwen38_flash_next.native_cpu COMMAND qwen38_flash_next_native_cpu_test)
set_tests_properties(qwen38_flash_next.native_cpu PROPERTIES LABELS "cpu;qwen38_flash_next"
  SKIP_RETURN_CODE 77)

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
  add_executable(qwen38_flash_next_ple_rows_bench EXCLUDE_FROM_ALL
    ple_rows_bench.cpp expert_bench_cache.hip.cpp)
  target_link_libraries(qwen38_flash_next_ple_rows_bench PRIVATE
    gufo_qwen38_flash_next hip::host)
  add_executable(qwen38_flash_next_native_bench EXCLUDE_FROM_ALL native_bench.cpp)
  target_link_libraries(qwen38_flash_next_native_bench PRIVATE gufo_qwen38_flash_next)
  foreach(name IN ITEMS mtp_ops blaslt_ops hc_mix_ops gdn_ops
      attention_ops kv_q8_ops moe_epilogue_ops moe_ids_ops projection_ops routed_wmma_ops
      iq_experts_ops
      select_ops prefill_experts_ops stageb_ops snapshot_copy verification_sampling)
    add_qwen38_flash_next_test(${name} gufo_qwen38_flash_next "hip;gpu;gfx1151")
  endforeach()
  add_qwen38_flash_next_test(kv_rotation_ops gufo_qwen38_flash_next "hip;gpu;gfx1151")
  target_sources(qwen38_flash_next_kv_rotation_ops_test PRIVATE
    kv_rotation_producer_native.cpp)
  target_compile_options(qwen38_flash_next_kv_rotation_ops_test PRIVATE
    -fno-fast-math -ffp-contract=off -frounding-math)
  # Compile the actual private q8 load helper under production HIP flags;
  # exhaust its valid finite scale/code domain against the portable codec.
  add_executable(qwen38_flash_next_kv_q8_load_test kv_q8_load_test.hip)
  target_link_libraries(qwen38_flash_next_kv_q8_load_test PRIVATE
    gufo_qwen38_flash_next hip::host)
  target_compile_options(qwen38_flash_next_kv_q8_load_test PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3;-ffast-math;-fno-finite-math-only>)
  add_test(NAME qwen38_flash_next.kv_q8_load
    COMMAND qwen38_flash_next_kv_q8_load_test)
  set_tests_properties(qwen38_flash_next.kv_q8_load PROPERTIES
    LABELS "qwen38_flash_next;hip;gpu;gfx1151" TIMEOUT 60)
  add_dependencies(qwen38_flash_next_tests qwen38_flash_next_kv_q8_load_test)

  # The rope phase and RMS checks compare against the scalar CPU oracle.
  target_link_libraries(qwen38_flash_next_attention_ops_test PRIVATE
    gufo_qwen38_flash_next_reference)

  add_executable(qwen38_flash_next_session_test EXCLUDE_FROM_ALL session_test.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/serve/continuation_cache.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/serve/continuation_disk_store.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/serve/component_text_cache.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/serve/text_model_runner.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/serve/text_generation_scheduler.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/serve/inference_backend.cpp)
  # inference_backend.cpp also serves Gemma 4 since the Gemma merge
  target_link_libraries(qwen38_flash_next_session_test PRIVATE gufo_logging
    gufo_qwen38_flash_next gufo_deepseek_v4_flash gufo_gemma4 gufo_cache_hip hip::host)
  set(QFN_RESET_FAULT_SOURCE reset_submission_fault.cpp)
  if(WIN32)
    set(QFN_RESET_FAULT_SOURCE reset_submission_fault_windows.cpp)
  endif()
  add_executable(qwen38_flash_next_continuation_adapter_test EXCLUDE_FROM_ALL
    continuation_adapter_test.cpp ${QFN_RESET_FAULT_SOURCE})
  target_link_libraries(qwen38_flash_next_continuation_adapter_test PRIVATE
    gufo_qwen38_flash_next ${CMAKE_DL_LIBS})
  add_executable(qwen38_flash_next_continuation_probe EXCLUDE_FROM_ALL
    continuation_probe.cpp)
  target_link_libraries(qwen38_flash_next_continuation_probe PRIVATE
    gufo_qwen38_flash_next)
  add_executable(qwen38_flash_next_snapshot_test EXCLUDE_FROM_ALL snapshot_test.cpp
    device_headroom_fault.cpp yarn_producer_native.cpp)
  target_link_libraries(qwen38_flash_next_snapshot_test PRIVATE
    gufo_qwen38_flash_next ${CMAKE_DL_LIBS})
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
    DEPENDS qwen38_flash_next_session_test qwen38_flash_next_snapshot_test
      qwen38_flash_next_continuation_adapter_test)

  set_source_files_properties(dense_gemm_bench_dequant.hip.cpp PROPERTIES LANGUAGE HIP)
  add_executable(qwen38_flash_next_dense_gemm_bench EXCLUDE_FROM_ALL
    dense_gemm_bench.cpp dense_gemm_bench_dequant.hip.cpp)
  target_link_libraries(qwen38_flash_next_dense_gemm_bench PRIVATE
    gufo_qwen38_flash_next gufo_gemma4 hip::host)

  add_executable(qwen38_flash_next_dense_twin_gpu_test EXCLUDE_FROM_ALL dense_twin_gpu_test.cpp)
  target_link_libraries(qwen38_flash_next_dense_twin_gpu_test PRIVATE
    gufo_qwen38_flash_next hip::host)

  set_source_files_properties(expert_bench_cache.hip.cpp PROPERTIES LANGUAGE HIP)
  # tubo_k4*.hip.cpp: copied MIT dot4 kernels, ablation controls only. One
  # width per translation unit so the large unrolled kernels build in parallel.
  set(tubo_control_sources tubo_k4.hip.cpp tubo_k4_w1.hip.cpp
    tubo_k4_w2.hip.cpp tubo_k4_w4.hip.cpp tubo_k4_w8.hip.cpp)
  set_source_files_properties(${tubo_control_sources} PROPERTIES LANGUAGE HIP)
  add_executable(qwen38_flash_next_dense_gemv_bench EXCLUDE_FROM_ALL
    dense_gemv_bench.cpp expert_bench_cache.hip.cpp ${tubo_control_sources})
  target_link_libraries(qwen38_flash_next_dense_gemv_bench PRIVATE
    gufo_qwen38_flash_next hip::host)
  add_executable(qwen38_flash_next_expert_decode_bench EXCLUDE_FROM_ALL
    expert_decode_bench.cpp expert_bench_cache.hip.cpp)
  target_link_libraries(qwen38_flash_next_expert_decode_bench PRIVATE
    gufo_qwen38_flash_next hip::host)

  add_executable(qwen38_flash_next_head_two_bench EXCLUDE_FROM_ALL
    head_two_bench.cpp expert_bench_cache.hip.cpp)
  target_link_libraries(qwen38_flash_next_head_two_bench PRIVATE
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
