add_executable(memory_connection_test "${CMAKE_SOURCE_DIR}/tests/cli/contract/memory_connection_test.cpp")
target_link_libraries(memory_connection_test PRIVATE bubo_prepare Threads::Threads)
add_test(NAME memory_connection_test COMMAND memory_connection_test)
set_tests_properties(memory_connection_test PROPERTIES LABELS "cpu;cli;contract;memory" TIMEOUT 30)
foreach(variant npu no_npu)
  add_executable(settings_${variant}_test
    "${CMAKE_SOURCE_DIR}/tests/cli/contract/settings_test.cpp"
    "${CMAKE_SOURCE_DIR}/src/cli/serve/release_profile.cpp"
    "${CMAKE_SOURCE_DIR}/src/cli/serve/model_manifest.cpp")
  target_link_libraries(settings_${variant}_test PRIVATE gufo_core gufo_logging bubo_prepare)
  if(variant STREQUAL "npu")
    target_compile_definitions(settings_${variant}_test PRIVATE GUFO_NPU=1)
  endif()
  add_test(NAME settings_${variant}_test COMMAND settings_${variant}_test)
  set_tests_properties(settings_${variant}_test PROPERTIES LABELS "cpu;cli;contract;settings" TIMEOUT 30)
endforeach()
add_executable(models_test "${CMAKE_SOURCE_DIR}/tests/cli/contract/models_test.cpp")
target_link_libraries(models_test PRIVATE bubo_models bubo_prepare gufo_core gufo_logging Threads::Threads)
add_test(NAME models_test COMMAND models_test)
set_tests_properties(models_test PROPERTIES LABELS "cpu;cli;contract;models" TIMEOUT 120)
add_test(NAME models_measured_admission_test COMMAND models_test --measured-data
  "${CMAKE_SOURCE_DIR}/tests/cli/contract/data/flash-next-admission.json")
set_tests_properties(models_measured_admission_test PROPERTIES LABELS "cpu;cli;contract;memory" TIMEOUT 30)
add_executable(memory96_probe_driver "${CMAKE_SOURCE_DIR}/tests/cli/contract/memory96_probe_driver.cpp")
target_link_libraries(memory96_probe_driver PRIVATE bubo_models bubo_prepare gufo_core gufo_logging Threads::Threads)
target_compile_definitions(memory96_probe_driver PRIVATE BUBO_MEMORY96_DATA_DIR="${CMAKE_SOURCE_DIR}/tests/cli/contract/data")
add_test(NAME memory96_process_test COMMAND memory96_probe_driver 96-64 self-test "${CMAKE_BINARY_DIR}/memory96-empty-models")
set_tests_properties(memory96_process_test PROPERTIES LABELS "cpu;cli;contract;memory;derived-probe" TIMEOUT 60)
add_executable(machine_adapter_test "${CMAKE_SOURCE_DIR}/tests/cli/contract/machine_adapter_test.cpp")
target_link_libraries(machine_adapter_test PRIVATE bubo_prepare Threads::Threads)
add_test(NAME machine_adapter_test COMMAND machine_adapter_test)
set_tests_properties(machine_adapter_test PROPERTIES LABELS "cpu;cli;contract;machine" TIMEOUT 60)
add_executable(machine_warnings_test "${CMAKE_SOURCE_DIR}/tests/cli/contract/machine_warnings_test.cpp")
target_link_libraries(machine_warnings_test PRIVATE bubo_prepare gufo_diagnostics Threads::Threads)
add_test(NAME machine_warnings_test COMMAND machine_warnings_test)
set_tests_properties(machine_warnings_test PROPERTIES LABELS "cpu;cli;contract;machine" TIMEOUT 60)
add_executable(bundle_contract_test "${CMAKE_SOURCE_DIR}/tests/cli/contract/bundle_test.cpp")
target_link_libraries(bundle_contract_test PRIVATE bubo_prepare)
add_test(NAME bundle_contract_test COMMAND bundle_contract_test)
set_tests_properties(bundle_contract_test PROPERTIES LABELS "cpu;cli;contract;privacy" TIMEOUT 30 WORKING_DIRECTORY "${CMAKE_BINARY_DIR}")
add_test(NAME switches_registry_test COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/cli/contract/switches_registry_test.py")
set_tests_properties(switches_registry_test PROPERTIES LABELS "cpu;cli;contract;registry" TIMEOUT 30)
add_test(NAME switches_recipe_test COMMAND "${Python3_EXECUTABLE}"
  "${CMAKE_SOURCE_DIR}/tests/cli/contract/switches_recipe_test.py" "$<TARGET_FILE:release_profile_test>")
set_tests_properties(switches_recipe_test PROPERTIES LABELS "cpu;cli;contract;registry" TIMEOUT 60)
if(WIN32 AND GUFO_BRAND STREQUAL "bubo")
  add_test(NAME bundle_process_test COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_SOURCE_DIR}/tests/cli/contract/bundle_process_test.py"
    "$<TARGET_FILE:status_bootstrap_test>" --work-root "${CMAKE_BINARY_DIR}/contract-bundle-process")
  set_tests_properties(bundle_process_test PROPERTIES LABELS "cpu;cli;contract;privacy" TIMEOUT 60)
  add_test(NAME machine_process_test COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_SOURCE_DIR}/tests/cli/contract/machine_process_test.py"
    "$<TARGET_FILE:status_bootstrap_test>" --work-root "${CMAKE_BINARY_DIR}/contract-machine-process")
  set_tests_properties(machine_process_test PROPERTIES LABELS "cpu;cli;contract;machine" TIMEOUT 60)
  add_test(NAME helpers_process_test COMMAND "${contract_pwsh}" -NoProfile -File
    "${CMAKE_SOURCE_DIR}/tests/cli/contract/helpers_process_test.ps1"
    -Bootstrap "$<TARGET_FILE:status_bootstrap_test>" -MemoryTest "$<TARGET_FILE:memory_connection_test>")
  set_tests_properties(helpers_process_test PROPERTIES LABELS "cpu;cli;contract;connection" TIMEOUT 60)
endif()
add_test(NAME machine_v1_compatibility_test COMMAND "${contract_pwsh}" -NoProfile -File
  "${CMAKE_SOURCE_DIR}/tests/cli/contract/compatibility_test.ps1")
set_tests_properties(machine_v1_compatibility_test PROPERTIES LABELS "cpu;cli;contract;schema" TIMEOUT 60)
