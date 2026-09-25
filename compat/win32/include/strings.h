/* Win32 shadow of <strings.h>. */
#pragma once
#include <string.h>
#ifndef strcasecmp
#define strcasecmp _stricmp
#define strncasecmp _strnicmp
#endif
