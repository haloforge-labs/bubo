if(CONFIG STREQUAL "Release")
  execute_process(COMMAND "${GATE}" --release "${EVIDENCE}"
    RESULT_VARIABLE result)
  if(NOT result EQUAL 0)
    message(FATAL_ERROR "Prepare production pins/evidence are not release-ready")
  endif()
endif()
