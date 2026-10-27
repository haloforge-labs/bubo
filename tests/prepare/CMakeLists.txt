# The real Windows CLI uses the OS memory guard even in a CPU-only test
# build. Production adds these sources in its HIP branch; CPU process tests
# need the same definitions to link bubo-engine.dll.
if(WIN32 AND NOT ENGINE_ENABLE_HIP)
  target_sources(gufo_core PRIVATE
    "${CMAKE_SOURCE_DIR}/src/core/platform/memory_readings.cpp")
endif()

add_executable(machine_writer_test ../cli/contract/writer_test.cpp)
add_executable(argparse_driver ../cli/contract/argparse_driver.cpp)
target_link_libraries(argparse_driver PRIVATE bubo_models gufo_diagnostics)
add_executable(command_table_test ../cli/command_table_test.cpp)
target_link_libraries(command_table_test PRIVATE bubo_prepare)
add_test(NAME command_table_test COMMAND command_table_test)
set_tests_properties(command_table_test PROPERTIES LABELS "cpu;cli;contract")
if(GUFO_BRAND STREQUAL "bubo")
  # Process records freeze the Windows bootstrap/local-control contract.
  # Pure parser fixtures also run on Linux, with generated paths normalized.
  set(argparse_golden_scope)
  if(NOT WIN32)
    set(argparse_golden_scope --only pure)
  endif()
  add_test(NAME argparse_golden_test COMMAND ${Python3_EXECUTABLE}
    "${CMAKE_SOURCE_DIR}/tests/cli/contract/argparse_golden_test.py"
    --driver "$<TARGET_FILE:argparse_driver>" --cli "$<TARGET_FILE:gufo>"
    ${argparse_golden_scope})
  set_tests_properties(argparse_golden_test PROPERTIES LABELS "cpu;cli;contract" TIMEOUT 90)
endif()
include("${CMAKE_SOURCE_DIR}/tests/cli/contract/serve_control_targets.cmake")
include("${CMAKE_SOURCE_DIR}/tests/cli/contract/settings_support_targets.cmake")
add_executable(prepared_default_test ../cli/contract/resolver_test.cpp)
target_link_libraries(prepared_default_test PRIVATE bubo_prepare)
add_test(NAME prepared_default_test COMMAND prepared_default_test "${CMAKE_BINARY_DIR}/contract-resolver")
set_tests_properties(prepared_default_test PROPERTIES LABELS "cpu;cli;contract;resolver")
if(WIN32 AND GUFO_BRAND STREQUAL "bubo")
  # The real production bootstrap sources and library, without an engine
  # dependency. This target must work from a folder with no AMD runtime.
  add_executable(status_bootstrap_test ../../src/cli/bootstrap.cpp)
  target_link_libraries(status_bootstrap_test PRIVATE bubo_prepare)
  set_target_properties(status_bootstrap_test PROPERTIES OUTPUT_NAME "bubo"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/contract-bootstrap")
  add_test(NAME status_bootstrap_test COMMAND ${Python3_EXECUTABLE}
    "${CMAKE_SOURCE_DIR}/tests/cli/contract/status_process_test.py"
    "$<TARGET_FILE:status_bootstrap_test>" "${CMAKE_BINARY_DIR}/contract-status-empty")
  set_tests_properties(status_bootstrap_test PROPERTIES LABELS "cpu;cli;contract")
endif()
add_executable(diagnose_contract_test ../cli/contract/diagnose_test.cpp)
add_executable(windows_inventory_test ../cli/contract/windows_inventory_test.cpp)
target_link_libraries(windows_inventory_test PRIVATE gufo_diagnostics)
add_test(NAME windows_inventory_test COMMAND windows_inventory_test)
set_tests_properties(windows_inventory_test PROPERTIES LABELS "cpu;cli;contract")
target_link_libraries(diagnose_contract_test PRIVATE gufo_diagnostics)
add_executable(machine_exit_test ../cli/contract/exit_test.cpp)
target_link_libraries(machine_exit_test PRIVATE bubo_prepare)
add_test(NAME machine_exit_test COMMAND machine_exit_test)
set_tests_properties(machine_exit_test PROPERTIES LABELS "cpu;cli;contract")
target_link_libraries(machine_writer_test PRIVATE bubo_prepare)
add_test(NAME machine_writer_test COMMAND machine_writer_test)
add_test(NAME machine_writer_process_test COMMAND ${Python3_EXECUTABLE}
  "${CMAKE_SOURCE_DIR}/tests/cli/contract/writer_process_test.py" "$<TARGET_FILE:machine_writer_test>")
set_tests_properties(machine_writer_process_test PROPERTIES LABELS "cpu;cli;contract")
set_tests_properties(machine_writer_test PROPERTIES LABELS "cpu;cli;contract")
add_test(NAME machine_writer_fault_test COMMAND ${Python3_EXECUTABLE}
  "${CMAKE_SOURCE_DIR}/tests/cli/contract/writer_fault_test.py" "$<TARGET_FILE:machine_writer_test>" "$<TARGET_FILE:prepare_input_driver>")
set_tests_properties(machine_writer_fault_test PROPERTIES LABELS "cpu;cli;contract" TIMEOUT 30)
add_executable(prepare_input_driver ../cli/contract/prepare_input_driver.cpp)
target_link_libraries(prepare_input_driver PRIVATE bubo_prepare)
add_test(NAME machine_input_pipe_fault_test COMMAND ${Python3_EXECUTABLE}
  "${CMAKE_SOURCE_DIR}/tests/cli/contract/input_pipe_test.py" "$<TARGET_FILE:prepare_input_driver>")
set_tests_properties(machine_input_pipe_fault_test PROPERTIES LABELS "cpu;cli;contract" TIMEOUT 120)
add_executable(prerequisite_fault_test ../cli/contract/prerequisite_fault_test.cpp)
target_link_libraries(prerequisite_fault_test PRIVATE bubo_prepare)
add_test(NAME prerequisite_fault_test COMMAND prerequisite_fault_test)
set_tests_properties(prerequisite_fault_test PROPERTIES LABELS "cpu;cli;contract")
add_test(NAME machine_contract_docs_test COMMAND ${Python3_EXECUTABLE}
  "${CMAKE_SOURCE_DIR}/tests/cli/contract/check_docs.py")
set_tests_properties(machine_contract_docs_test PROPERTIES LABELS "cpu;cli;contract")
if(WIN32 AND GUFO_BRAND STREQUAL "bubo")
  add_test(NAME machine_loader_fault_test COMMAND ${Python3_EXECUTABLE}
    "${CMAKE_SOURCE_DIR}/tests/cli/contract/loader_fault_test.py"
    "$<TARGET_FILE:status_bootstrap_test>" "${CMAKE_BINARY_DIR}/contract-loader-missing-crt")
  set_tests_properties(machine_loader_fault_test PROPERTIES LABELS "cpu;cli;contract" TIMEOUT 30)
endif()
find_program(contract_pwsh NAMES pwsh)
if(contract_pwsh)
  add_test(NAME machine_docs_schema_test COMMAND ${contract_pwsh} -NoProfile -File
    "${CMAKE_SOURCE_DIR}/tests/cli/contract/docs_schema_test.ps1")
  set_tests_properties(machine_docs_schema_test PROPERTIES LABELS "cpu;cli;contract")
  add_test(NAME diagnose_contract_test COMMAND ${contract_pwsh} -NoProfile -File
    "${CMAKE_SOURCE_DIR}/tests/cli/contract/diagnose_process_test.ps1"
    "-Test" "$<TARGET_FILE:diagnose_contract_test>" "-Build" "${CMAKE_BINARY_DIR}")
  set_tests_properties(diagnose_contract_test PROPERTIES LABELS "cpu;cli;contract")
  add_test(NAME machine_schema_test COMMAND ${contract_pwsh} -NoProfile -File
    "${CMAKE_SOURCE_DIR}/tests/cli/contract/schema_test.ps1")
  set_tests_properties(machine_schema_test PROPERTIES LABELS "cpu;cli;contract")
endif()
foreach(name prepare_vc_runtime_test prepare_args_test prepare_catalogue_test prepare_gate_test
             prepare_fingerprint_test prepare_dependencies_test prepare_zip_test
             runtime_location_test)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} PRIVATE bubo_prepare)
  set_target_properties(${name} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
  target_compile_definitions(${name} PRIVATE
    BUBO_PREPARE_FIXTURES="${CMAKE_CURRENT_SOURCE_DIR}/fixtures")
  add_test(NAME ${name} COMMAND ${name})
  set_tests_properties(${name} PROPERTIES LABELS "cpu;prepare;release")
endforeach()
list(TRANSFORM bubo_prepare_sources PREPEND "${CMAKE_SOURCE_DIR}/"
  OUTPUT_VARIABLE bubo_prepare_test_sources)
add_library(bubo_prepare_test STATIC ${bubo_prepare_test_sources})
add_executable(prepare_machine_test ../cli/contract/prepare_machine_test.cpp)
target_link_libraries(prepare_machine_test PRIVATE bubo_prepare_test)
add_executable(prepare_fault_test ../cli/contract/prepare_fault_test.cpp)
target_link_libraries(prepare_fault_test PRIVATE bubo_prepare_test)
if(contract_pwsh)
  add_test(NAME prepare_fault_test COMMAND ${contract_pwsh} -NoProfile -File
    "${CMAKE_SOURCE_DIR}/tests/cli/contract/prepare_process_test.ps1"
    "-Test" "$<TARGET_FILE:prepare_fault_test>" "-Build" "${CMAKE_BINARY_DIR}")
  set_tests_properties(prepare_fault_test PROPERTIES LABELS "cpu;prepare;contract" TIMEOUT 120)
  add_test(NAME prepare_machine_test COMMAND ${contract_pwsh} -NoProfile -File
    "${CMAKE_SOURCE_DIR}/tests/cli/contract/prepare_process_test.ps1"
    "-Test" "$<TARGET_FILE:prepare_machine_test>" "-Build" "${CMAKE_BINARY_DIR}")
  set_tests_properties(prepare_machine_test PROPERTIES LABELS "cpu;prepare;contract")
endif()
target_link_libraries(bubo_prepare_test PUBLIC CURL::libcurl OpenSSL::Crypto ZLIB::ZLIB)
target_compile_definitions(bubo_prepare_test PUBLIC BUBO_PREPARE_TESTING=1)
if(WIN32)
  target_link_libraries(bubo_prepare_test PUBLIC Rstrtmgr Advapi32 dxgi)
endif()
add_executable(prepare_runtime_test prepare_runtime_test.cpp loopback_stub.cpp)
target_link_libraries(prepare_runtime_test PRIVATE bubo_prepare_test)
target_compile_definitions(prepare_runtime_test PRIVATE
  BUBO_PREPARE_FIXTURES="${CMAKE_CURRENT_SOURCE_DIR}/fixtures"
  BUBO_NOTICE_SOURCE="${CMAKE_SOURCE_DIR}/tools/release/runtime-notices")
set_target_properties(prepare_runtime_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
if(WIN32)
  target_link_libraries(prepare_runtime_test PRIVATE ws2_32)
  add_dependencies(prepare_runtime_test prepare_fixture_engine)
  target_compile_definitions(prepare_runtime_test PRIVATE BUBO_INERT_DLL="$<TARGET_FILE:prepare_fixture_engine>")
endif()
add_test(NAME prepare_runtime_test COMMAND prepare_runtime_test
  "${CMAKE_BINARY_DIR}/j3-ctest-runtime" "${CMAKE_CURRENT_SOURCE_DIR}/fixtures/runtime-consent-source.md")
set_tests_properties(prepare_runtime_test PROPERTIES LABELS "cpu;prepare;release")
include(runtime_mutations.cmake)
foreach(name prepare_transport_test prepare_retry_policy_test prepare_large_resume_test prepare_model_test prepare_loopback_test)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} PRIVATE bubo_prepare_test)
  target_compile_definitions(${name} PRIVATE
    BUBO_PREPARE_FIXTURES="${CMAKE_CURRENT_SOURCE_DIR}/fixtures")
  set_target_properties(${name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
  if(name STREQUAL "prepare_model_test" OR name STREQUAL "prepare_loopback_test")
    if(name STREQUAL "prepare_model_test")
      set(kind model)
    else()
      set(kind loopback)
    endif()
    add_test(NAME ${name} COMMAND ${CMAKE_COMMAND}
      "-DTEST=$<TARGET_FILE:${name}>" "-DBUILD=${CMAKE_BINARY_DIR}"
      "-DKIND=${kind}" "-DGOLDEN=${CMAKE_CURRENT_SOURCE_DIR}/fixtures/model-consent.txt"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/fixture_test.cmake")
  else()
    add_test(NAME ${name} COMMAND ${name})
  endif()
  set_tests_properties(${name} PROPERTIES LABELS "cpu;prepare;release")
endforeach()
target_sources(prepare_loopback_test PRIVATE loopback_stub.cpp)
target_sources(prepare_large_resume_test PRIVATE loopback_stub.cpp)
target_sources(prepare_model_test PRIVATE loopback_stub.cpp)
if(WIN32)
  target_link_libraries(prepare_loopback_test PRIVATE ws2_32)
  target_link_libraries(prepare_large_resume_test PRIVATE ws2_32)
  target_link_libraries(prepare_model_test PRIVATE ws2_32)
endif()
add_executable(prepare_model_set_test prepare_model_set_test.cpp
  ../../src/cli/serve/model_manifest.cpp ../../src/cli/serve/release_profile.cpp)
target_link_libraries(prepare_model_set_test PRIVATE bubo_prepare_test gufo_core gufo_logging)
target_compile_definitions(prepare_model_set_test PRIVATE
  BUBO_PREPARE_FIXTURES="${CMAKE_CURRENT_SOURCE_DIR}/fixtures")
set_target_properties(prepare_model_set_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
add_test(NAME prepare_model_set_test COMMAND prepare_model_set_test)
set_tests_properties(prepare_model_set_test PROPERTIES LABELS "cpu;prepare;release")
add_executable(model_catalogue_test model_catalogue_test.cpp)
target_link_libraries(model_catalogue_test PRIVATE bubo_prepare_test)
target_compile_definitions(model_catalogue_test PRIVATE
  BUBO_PREPARE_FIXTURES="${CMAKE_CURRENT_SOURCE_DIR}/fixtures")
set_target_properties(model_catalogue_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
add_test(NAME model_catalogue_test COMMAND model_catalogue_test)
set_tests_properties(model_catalogue_test PROPERTIES LABELS "cpu;prepare;release;catalogue" TIMEOUT 120)
# The embedded notice texts are generated files: the check regenerates each set
# from tools/release/model-notices and compares (exit 1 on any difference).
find_package(Python3 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
  add_test(NAME model_notices_generated_huihui COMMAND ${Python3_EXECUTABLE} -I
    "${CMAKE_SOURCE_DIR}/tools/release/gen_model_notices.py" --check
    "${CMAKE_SOURCE_DIR}/tools/release/model-notices/huihui"
    "${CMAKE_SOURCE_DIR}/src/core/prepare/model_notices.inc")
  add_test(NAME model_notices_generated_original COMMAND ${Python3_EXECUTABLE} -I
    "${CMAKE_SOURCE_DIR}/tools/release/gen_model_notices.py" --check
    "${CMAKE_SOURCE_DIR}/tools/release/model-notices/original"
    "${CMAKE_SOURCE_DIR}/src/core/prepare/model_notices_original.inc" kOriginalModelNotices)
  add_test(NAME model_notices_generated_lowmemory COMMAND ${Python3_EXECUTABLE} -I
    "${CMAKE_SOURCE_DIR}/tools/release/gen_model_notices.py" --check
    "${CMAKE_SOURCE_DIR}/tools/release/model-notices/lowmemory"
    "${CMAKE_SOURCE_DIR}/src/core/prepare/model_notices_lowmemory.inc" kLowMemoryModelNotices)
  set_tests_properties(model_notices_generated_huihui model_notices_generated_original model_notices_generated_lowmemory
    PROPERTIES LABELS "cpu;prepare;release")
endif()
foreach(name prepare_policy_test prepare_gguf_fingerprint_test)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} PRIVATE bubo_prepare)
  set_target_properties(${name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
  add_test(NAME ${name} COMMAND ${name})
  set_tests_properties(${name} PROPERTIES LABELS "cpu;prepare;release")
endforeach()
include(model_mutations.cmake)
include(transport_mutations.cmake)
add_test(NAME prepare_help_snapshot
  COMMAND ${CMAKE_COMMAND} "-DBUBO=$<TARGET_FILE:gufo>"
    "-DGOLDEN=${CMAKE_CURRENT_SOURCE_DIR}/fixtures/prepare-help.txt"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/help_snapshot.cmake")
set_tests_properties(prepare_help_snapshot PROPERTIES LABELS "cpu;prepare;cli")
if(WIN32 AND GUFO_BRAND STREQUAL "bubo")
  add_library(prepare_fixture_engine SHARED bootstrap_fixture_engine.cpp)
  set_target_properties(prepare_fixture_engine PROPERTIES PREFIX ""
    OUTPUT_NAME "bubo-engine"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/prepare-fixture-engine")
endif()

add_executable(model_manifest_test
  ../serve/model_manifest_test.cpp
  ../../src/cli/serve/model_manifest.cpp
  ../../src/cli/serve/release_profile.cpp)
target_link_libraries(model_manifest_test PRIVATE bubo_prepare_test gufo_core gufo_logging)
set_target_properties(model_manifest_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
add_test(NAME model_manifest_test COMMAND model_manifest_test)
set_tests_properties(model_manifest_test PROPERTIES LABELS "cpu;prepare;release;resolver")

# Mutated implementations must fail the intact manifest/cache identity cases.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${CMAKE_SOURCE_DIR}/src/cli/serve/model_manifest.cpp")
file(READ "${CMAKE_SOURCE_DIR}/src/cli/serve/model_manifest.cpp" manifest_source)
foreach(kind stamp_match automatic_hash)
  if(kind STREQUAL "stamp_match")
    set(anchor "stored != stamps.end() && stored->second == stamp")
    set(replacement "stored != stamps.end()")
  else()
    set(anchor "Verify(*file, package, output);")
    set(replacement "(void)file;")
  endif()
  string(FIND "${manifest_source}" "${anchor}" found)
  if(found EQUAL -1)
    message(FATAL_ERROR "manifest ${kind} mutation anchor missing")
  endif()
  string(REPLACE "${anchor}" "${replacement}" mutated "${manifest_source}")
  set(copy "${CMAKE_CURRENT_BINARY_DIR}/manifest_mutation_${kind}.cpp")
  file(WRITE "${copy}" "${mutated}")
  set(name "model_manifest_mutant_${kind}")
  add_executable(${name} ../serve/model_manifest_test.cpp "${copy}"
    ../../src/cli/serve/release_profile.cpp)
  target_link_libraries(${name} PRIVATE bubo_prepare_test gufo_core gufo_logging)
  set_target_properties(${name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
  add_test(NAME ${name} COMMAND ${name})
  set_tests_properties(${name} PROPERTIES WILL_FAIL TRUE
    LABELS "cpu;prepare;release;resolver;mutation")
endforeach()

add_executable(prepare_integration_test prepare_integration_test.cpp
  ../../src/cli/serve/model_manifest.cpp ../../src/cli/serve/release_profile.cpp)
target_link_libraries(prepare_integration_test PRIVATE bubo_prepare_test gufo_core gufo_logging)
set_target_properties(prepare_integration_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
add_test(NAME prepare_integration_test COMMAND prepare_integration_test)
set_tests_properties(prepare_integration_test PROPERTIES LABELS "cpu;prepare;release;integration")
add_custom_target(prepare_ci_test
  COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/prepare_ci_test.py"
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}" VERBATIM)
add_test(NAME prepare_ci_test
  COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/prepare_ci_test.py")
set_tests_properties(prepare_ci_test PROPERTIES WORKING_DIRECTORY "${CMAKE_BINARY_DIR}" LABELS "cpu;prepare;release;ci")
include(vc_runtime_mutations.cmake)
if(WIN32)
  find_program(BUBO_PWSH pwsh REQUIRED)
  foreach(script package_crt_test vc_runtime_launcher_test)
    add_test(NAME ${script} COMMAND "${BUBO_PWSH}" -NoProfile -File
      "${CMAKE_CURRENT_SOURCE_DIR}/${script}.ps1")
    set_tests_properties(${script} PROPERTIES WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
      LABELS "cpu;prepare;release")
  endforeach()
endif()
add_executable(prepare_plan_driver EXCLUDE_FROM_ALL prepare_plan_driver.cpp)
target_link_libraries(prepare_plan_driver PRIVATE bubo_prepare)
set_target_properties(prepare_plan_driver PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
