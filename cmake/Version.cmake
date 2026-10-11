# Tracked versions are read on every configure. Retired -D cache entries and
# normal variables cannot override them, including in reused build folders.
get_filename_component(version_root "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
unset(GUFO_RELEASE_VERSION CACHE)
unset(GUFO_PRODUCT_VERSION CACHE)
unset(GUFO_REVISION CACHE)
file(READ "${version_root}/version.txt" GUFO_RELEASE_VERSION)
file(READ "${version_root}/tools/release/product_version.txt" GUFO_PRODUCT_VERSION)
string(STRIP "${GUFO_RELEASE_VERSION}" GUFO_RELEASE_VERSION)
string(STRIP "${GUFO_PRODUCT_VERSION}" GUFO_PRODUCT_VERSION)
foreach(version GUFO_RELEASE_VERSION GUFO_PRODUCT_VERSION)
  if(NOT "${${version}}" MATCHES "^[0-9]+\\.[0-9]+\\.[0-9]+$")
    message(FATAL_ERROR "${version} must contain a semantic base version")
  endif()
endforeach()
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${version_root}/version.txt" "${version_root}/tools/release/product_version.txt")

set(GUFO_REVISION "unknown")
find_package(Git QUIET)
if(GIT_FOUND)
  execute_process(COMMAND "${GIT_EXECUTABLE}" rev-parse --short HEAD
    WORKING_DIRECTORY "${version_root}"
    OUTPUT_VARIABLE git_revision OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET RESULT_VARIABLE git_result)
  if(git_result EQUAL 0 AND git_revision MATCHES "^[0-9a-f]+$")
    set(GUFO_REVISION "${git_revision}")
    execute_process(COMMAND "${GIT_EXECUTABLE}" symbolic-ref -q HEAD
      WORKING_DIRECTORY "${version_root}"
      OUTPUT_VARIABLE git_head_ref OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    foreach(ref HEAD "${git_head_ref}" packed-refs)
      if(ref STREQUAL "")
        continue()
      endif()
      execute_process(COMMAND "${GIT_EXECUTABLE}" rev-parse --git-path "${ref}"
        WORKING_DIRECTORY "${version_root}"
        OUTPUT_VARIABLE ref_path OUTPUT_STRIP_TRAILING_WHITESPACE)
      if(NOT IS_ABSOLUTE "${ref_path}")
        set(ref_path "${version_root}/${ref_path}")
      endif()
      set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${ref_path}")
    endforeach()
  endif()
endif()
set(GUFO_BUILD_REVISION "${GUFO_REVISION}")

function(gufo_escape_version_regex value output)
  string(REGEX REPLACE "([][+.*()^$?|\\\\])" "\\\\\\1" escaped "${value}")
  set(${output} "${escaped}" PARENT_SCOPE)
endfunction()
