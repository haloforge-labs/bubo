/* <windows.h> defines macros that collide with ordinary C++ identifiers in
 * Gufo (enumerators such as DELETE or ERROR, members such as SendMessage).
 * Include this after any Win32 header. Only names that break Gufo belong
 * here: removing a macro another Win32 header still needs breaks that header. */
#undef DELETE
#undef ERROR
#undef IN
#undef OUT
#undef OPTIONAL
#undef interface
#undef near
#undef far
#undef small
#undef min
#undef max
#undef SendMessage
#undef PostMessage
#undef GetMessage
#undef GetObject
#undef LoadImage
#undef DrawText
#undef GetCurrentTime
#undef Yield
