# Included by tests/prepare/CMakeLists.txt. These CPU targets do not depend on
# an inference model library, HIP or the production engine DLL. The driver
# links gufo_core's CPU metadata parser for real release-setting rejections.
find_program(contract_pwsh NAMES pwsh)
add_executable(cache_completion_test
  "${CMAKE_SOURCE_DIR}/tests/cli/contract/cache_completion_test.cpp")
target_compile_features(cache_completion_test PRIVATE cxx_std_20)
target_link_libraries(cache_completion_test PRIVATE Threads::Threads)
add_test(NAME cache_completion_test COMMAND cache_completion_test)
set_tests_properties(cache_completion_test PROPERTIES
  LABELS "cpu;cli;contract;server;metrics" TIMEOUT 30)

add_executable(control_unit_test
  "${CMAKE_SOURCE_DIR}/tests/cli/contract/control_unit_test.cpp")
target_link_libraries(control_unit_test PRIVATE bubo_prepare Threads::Threads)
add_test(NAME control_unit_test COMMAND control_unit_test)
set_tests_properties(control_unit_test PROPERTIES
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
  LABELS "cpu;cli;contract;control" TIMEOUT 30)

if(WIN32)
  add_executable(status_watch_replay
    "${CMAKE_SOURCE_DIR}/tests/cli/contract/status_watch_replay.cpp")
  target_link_libraries(status_watch_replay PRIVATE bubo_prepare Threads::Threads)
  add_test(NAME status_watch_process_test COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_SOURCE_DIR}/tests/cli/contract/status_watch_process_test.py"
    --cli "$<TARGET_FILE:gufo>" --bridge "$<TARGET_FILE:status_watch_replay>"
    --artifacts "${CMAKE_BINARY_DIR}/contract-status-watch")
  set_tests_properties(status_watch_process_test PROPERTIES
    LABELS "cpu;cli;contract;control" TIMEOUT 180)

  add_executable(npu_status_publisher_test
    "${CMAKE_SOURCE_DIR}/tests/cli/contract/npu_status_publisher_test.cpp"
    "${CMAKE_SOURCE_DIR}/src/npu/runtime_status.cpp")
  target_link_libraries(npu_status_publisher_test PRIVATE bubo_prepare Threads::Threads)
  add_test(NAME npu_status_publisher_test COMMAND npu_status_publisher_test)
  set_tests_properties(npu_status_publisher_test PROPERTIES
    LABELS "cpu;cli;contract;fake" TIMEOUT 30)

  # Non-text adapters and image acquisition are explicit unavailable fixtures
  # in the driver. The listener, text routes, local control, command wrapper,
  # discovery, metrics and log sink use their actual production source.
  # behaviour and text_sampling_defaults are header-only; the transport does
  # not call QwenChatTemplate/tokenizer implementations or load GGUF files.
  add_executable(serve_control_driver
    "${CMAKE_SOURCE_DIR}/tests/cli/contract/serve_control_driver.cpp"
    "${CMAKE_SOURCE_DIR}/src/npu/runtime_status.cpp"
    "${CMAKE_SOURCE_DIR}/src/cli/serve/release_profile.cpp"
    "${CMAKE_SOURCE_DIR}/src/cli/serve/model_manifest.cpp"
    "${CMAKE_SOURCE_DIR}/src/cli/serve/http_server.cpp"
    "${CMAKE_SOURCE_DIR}/src/cli/serve/openai_chat.cpp"
    "${CMAKE_SOURCE_DIR}/src/cli/serve/websocket.cpp"
    "${CMAKE_SOURCE_DIR}/src/models/gemma4/tool_syntax.cpp"
    "${CMAKE_SOURCE_DIR}/src/core/sampling.cpp"
    "${CMAKE_SOURCE_DIR}/src/core/json_constraint.cpp"
    "${CMAKE_SOURCE_DIR}/src/core/json_schema_lexeme.cpp"
    "${CMAKE_SOURCE_DIR}/src/core/json_schema_regex.cpp")
  target_compile_definitions(serve_control_driver PRIVATE BUBO_SERVE_TESTING=1)
  target_compile_features(serve_control_driver PRIVATE cxx_std_20)
  target_link_libraries(serve_control_driver PRIVATE
    bubo_prepare gufo_core gufo_logging Threads::Threads ICU::uc ICU::i18n)
  set_target_properties(serve_control_driver PROPERTIES
    OUTPUT_NAME "bubo"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/contract-serve-driver")
  add_test(NAME serve_control_process_test COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_SOURCE_DIR}/tests/cli/contract/serve_control_process_test.py"
    "$<TARGET_FILE:serve_control_driver>"
    "${CMAKE_BINARY_DIR}/contract-serve-control")
  set_tests_properties(serve_control_process_test PROPERTIES
    LABELS "cpu;cli;contract;server;control;fake" TIMEOUT 180)
endif()

if(contract_pwsh)
  add_test(NAME serve_test_symbol_gate COMMAND "${contract_pwsh}" -NoProfile -File
    "${CMAKE_SOURCE_DIR}/tests/cli/contract/serve_symbol_test.ps1"
    -BuildRoot "${CMAKE_BINARY_DIR}/contract-serve-symbols"
    -ProductionLibrary "$<TARGET_FILE:bubo_prepare>")
  set_tests_properties(serve_test_symbol_gate PROPERTIES
    LABELS "cpu;cli;contract;release;package" TIMEOUT 30)
endif()
