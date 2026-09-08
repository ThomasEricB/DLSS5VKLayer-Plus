// Own NVSDK_NGX_Parameter implementation with the verified non-standard
// 16-slot vtable layout (no virtual destructor on base).
// Ported from standalone_runner/main.cpp:134-246.
#pragma once
#include "ngx_abi.h"
#include "logging.h"
#include <map>
#include <set>
#include <string>

namespace dlssnr {

struct OwnParam final : NVSDK_NGX_Parameter {
    struct ParamVal {
        int kind = 0;  // 1=u64/int, 2=float, 3=double, 4=ptr
        unsigned long long u = 0;
        float f = 0.0f;
        double d = 0.0;
        void* p = nullptr;
    };
    mutable std::map<std::string, ParamVal> m;

    NVSDK_NGX_Result miss(const char* n) const {
        // Keyed per object, and the object is named.
        //
        // The set used to be shared by every parameter block in the process, which reports a key
        // once and then never again -- so a key set after its first miss went on looking missing
        // for the rest of the run, and two blocks could not be told apart at all. That cost two
        // rounds of chasing a contract that was already satisfied.
        if (n && logged.insert(n).second)
            Log("[param-miss] params=%p (%zu keys held) queried missing key: '%s'", (const void*)this, m.size(), n);
        return NVSDK_NGX_Result_FAIL_InvalidParameter;
    }
    mutable std::set<std::string> logged;

    // Slot 0 (0x00)
    void NVSDK_CONV Set(const char* n, void* v) override {
        if (!n) return;
        auto& x = m[n]; x = {}; x.kind = 4; x.p = v; x.u = (unsigned long long)v;
    }
    // Slot 1 (0x08)
    void NVSDK_CONV Set(const char* n, unsigned long long v) override {
        if (!n) return;
        auto& x = m[n]; x = {}; x.kind = 1; x.u = v; x.f = (float)v; x.d = (double)v;
    }
    // Slot 2 (0x10)
    void NVSDK_CONV Set(const char* n, float v) override {
        if (!n) return;
        auto& x = m[n]; x = {}; x.kind = 2; x.f = v; x.d = (double)v; x.u = (unsigned long long)v;
    }
    // Slot 3 (0x18)
    void NVSDK_CONV Set(const char* n, double v) override {
        if (!n) return;
        auto& x = m[n]; x = {}; x.kind = 3; x.d = v; x.f = (float)v; x.u = (unsigned long long)v;
    }
    // Slot 4 (0x20)
    void NVSDK_CONV Set(const char* n, unsigned int v) override {
        if (!n) return;
        auto& x = m[n]; x = {}; x.kind = 1; x.u = (unsigned long long)v; x.f = (float)v; x.d = (double)v;
    }
    // Slot 5 (0x28)
    void NVSDK_CONV Set(const char* n, int v) override {
        if (!n) return;
        auto& x = m[n]; x = {}; x.kind = 1; x.u = (unsigned long long)(unsigned int)v; x.f = (float)v; x.d = (double)v;
    }

    // Slot 6 (0x30)
    NVSDK_NGX_Result NVSDK_CONV Get(const char* n, double* v) const override {
        if (!n || !v) return NVSDK_NGX_Result_FAIL_InvalidParameter;
        auto it = m.find(n); if (it == m.end()) return miss(n);
        *v = (it->second.kind == 3) ? it->second.d : (it->second.kind == 2 ? (double)it->second.f : (double)it->second.u);
        if (Verbose()) Log("[param-get:double] '%s' -> %f", n, *v);
        return NVSDK_NGX_Result_Success;
    }
    // Slot 7 (0x38)
    NVSDK_NGX_Result NVSDK_CONV Get(const char* n, unsigned long long* v) const override {
        if (!n || !v) return NVSDK_NGX_Result_FAIL_InvalidParameter;
        auto it = m.find(n); if (it == m.end()) return miss(n);
        *v = it->second.u;
        if (Verbose()) Log("[param-get:ull] '%s' -> %llu", n, *v);
        return NVSDK_NGX_Result_Success;
    }
    // Slot 8 (0x40)
    NVSDK_NGX_Result NVSDK_CONV Get(const char* n, void** v) const override {
        if (!n || !v) return NVSDK_NGX_Result_FAIL_InvalidParameter;
        auto it = m.find(n); if (it == m.end()) { *v = nullptr; return miss(n); }
        *v = it->second.p ? it->second.p : (void*)(uintptr_t)it->second.u;
        if (Verbose()) Log("[param-get:void*] '%s' -> %p", n, *v);
        return NVSDK_NGX_Result_Success;
    }
    // Slot 9 (0x48)
    NVSDK_NGX_Result NVSDK_CONV GetReserved9(const char*, void*) const override {
        return NVSDK_NGX_Result_FAIL_InvalidParameter;
    }
    // Slot 10 (0x50)
    NVSDK_NGX_Result NVSDK_CONV GetReserved10(const char*, void*) const override {
        return NVSDK_NGX_Result_FAIL_InvalidParameter;
    }
    // Slot 11 (0x58)
    NVSDK_NGX_Result NVSDK_CONV Get(const char* n, int* v) const override {
        if (!n || !v) return NVSDK_NGX_Result_FAIL_InvalidParameter;
        auto it = m.find(n); if (it == m.end()) return miss(n);
        *v = (int)it->second.u;
        if (Verbose()) Log("[param-get:int] '%s' -> %d", n, *v);
        return NVSDK_NGX_Result_Success;
    }
    // Slot 12 (0x60)
    NVSDK_NGX_Result NVSDK_CONV Get(const char* n, unsigned int* v) const override {
        if (!n || !v) return NVSDK_NGX_Result_FAIL_InvalidParameter;
        auto it = m.find(n); if (it == m.end()) return miss(n);
        *v = (unsigned int)it->second.u;
        if (Verbose()) Log("[param-get:uint] '%s' -> %u", n, *v);
        return NVSDK_NGX_Result_Success;
    }
    // Slot 13 (0x68)
    NVSDK_NGX_Result NVSDK_CONV GetReserved13(const char*, void*) const override {
        return NVSDK_NGX_Result_FAIL_InvalidParameter;
    }
    // Slot 14 (0x70)
    NVSDK_NGX_Result NVSDK_CONV Get(const char* n, float* v) const override {
        if (!n || !v) return NVSDK_NGX_Result_FAIL_InvalidParameter;
        auto it = m.find(n); if (it == m.end()) return miss(n);
        *v = (it->second.kind == 2) ? it->second.f : (it->second.kind == 3 ? (float)it->second.d : (float)it->second.u);
        if (Verbose()) Log("[param-get:float] '%s' -> %f", n, *v);
        return NVSDK_NGX_Result_Success;
    }
    // Values that survive a reset.
    //
    // Frame generation calls Reset as the first thing it does inside evaluate, which cleared the
    // whole contract before it read a single key of it -- so every key reported missing while a
    // read-back a microsecond earlier said it was present. Anything marked to persist is put back,
    // which leaves the denoiser unaffected because it marks nothing.
    std::map<std::string, ParamVal> sticky;
    void Persist() { sticky = m; }

    // Slot 15 (0x78)
    void NVSDK_CONV Reset() override {
        Log("[param-reset] %zu keys cleared, %zu restored", m.size(), sticky.size());
        m = sticky;
    }
};

inline NVSDK_NGX_Result NVSDK_CONV ScalingRatioCallback(NVSDK_NGX_Parameter* parameters) noexcept {
    if (!parameters) return NVSDK_NGX_Result_FAIL_InvalidParameter;
    parameters->Set("DLSSNR.ScalingRatio", 1.0f);
    return NVSDK_NGX_Result_Success;
}

}  // namespace dlssnr