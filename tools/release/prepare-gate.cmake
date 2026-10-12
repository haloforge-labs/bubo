string(TOUPPER "${CONFIG}" prepare_config)
if(prepare_config STREQUAL "RELEASE")
  find_program(prepare_python NAMES python3 python REQUIRED)
  execute_process(COMMAND "${prepare_python}" "${ROOT}/tools/release/check_brand.py"
    --root "${ROOT}" --prepare-library "${LIBRARY}"
    RESULT_VARIABLE scan_result)
  if(NOT scan_result EQUAL 0)
    message(FATAL_ERROR "Prepare source/archive brand or security scan failed")
  endif()
  execute_process(COMMAND "${GATE}" --release "${EVIDENCE}"
    RESULT_VARIABLE result)
  if(NOT result EQUAL 0)
    message(FATAL_ERROR "Prepare production pins/evidence are not release-ready")
  endif()
endif()
