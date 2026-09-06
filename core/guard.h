#pragma once
#include <windows.h>
#include <csetjmp>
#include <cstdint>

namespace dlssnr {

// VEH + setjmp exception guard (MinGW has no __try/__except). Fail-closed:
// any exception inside Guarded() longjmps back and returns failValue.
void InstallGuard();
void RegisterModuleRange(const char* name, uintptr_t base, uintptr_t size);

extern thread_local jmp_buf g_guardJmp;
extern thread_local volatile bool g_guardActive;
extern thread_local volatile DWORD g_guardCode;
extern thread_local volatile int g_guardHits;

template <class R, class F>
R Guarded(F&& f, R failValue, DWORD* seh) {
    *seh = 0;
    g_guardHits = 0;
    if (setjmp(g_guardJmp) == 0) {
        g_guardActive = true;
        R result = f();
        g_guardActive = false;
        return result;
    }
    *seh = g_guardCode;
    return failValue;
}

}  // namespace dlssnr