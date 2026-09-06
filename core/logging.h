#pragma once
#include <windows.h>
#include <cstdio>
#include <cstdarg>

namespace dlssnr {

inline bool Verbose() {
    static const bool v = [] {
        char buf[8];
        return GetEnvironmentVariableA("DLSSNR_VERBOSE", buf, sizeof(buf)) > 0 && buf[0] == '1';
    }();
    return v;
}

inline void Log(const char* fmt, ...) {
    static FILE* f = [] {
        char path[MAX_PATH];
        if (GetEnvironmentVariableA("DLSSNR_LOG", path, MAX_PATH) == 0) return (FILE*)nullptr;
        return fopen(path, "a");
    }();
    char buf[2048];
    va_list args;
    va_start(args, fmt);
    _vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    char line[2200];
    _snprintf(line, sizeof(line), "[dlssnr] %s\r\n", buf);
    OutputDebugStringA(line);
    if (f) { fputs(line, f); fflush(f); }
}

}  // namespace dlssnr