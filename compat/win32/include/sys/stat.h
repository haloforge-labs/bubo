/* Win32 shadow of <sys/stat.h>; see gufo_posix.h. The UCRT header is parsed
 * completely before gufo_posix.h defines st_mtime and friends. */
#pragma once
#include_next <sys/stat.h>
#include "gufo_posix.h"
