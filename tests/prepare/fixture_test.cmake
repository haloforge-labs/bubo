string(RANDOM LENGTH 16 ALPHABET 0123456789abcdef run_id)
set(root "${BUILD}/prepare-fixture-${run_id}")
if(KIND STREQUAL "model")
  execute_process(COMMAND "${TEST}" "${root}" "${GOLDEN}"
    WORKING_DIRECTORY "${BUILD}" RESULT_VARIABLE result)
elseif(KIND STREQUAL "loopback")
  execute_process(COMMAND "${CMAKE_COMMAND}" -E env
    "BUBO_PREPARE_TEST_ROOT=${root}" "${TEST}"
    WORKING_DIRECTORY "${BUILD}" RESULT_VARIABLE result)
else()
  message(FATAL_ERROR "unknown fixture kind")
endif()
if(NOT result EQUAL 0)
  message(FATAL_ERROR "${KIND} fixture test failed: ${result}")
endif()
