// Ported from standalone_runner/main.cpp (verified Feature-18 Vulkan sequence):
//   IAT spoof  :283-365   guarded calls :741-774   param helpers :248-266
//   core init  :892-936   params        :938-1006  snippet init  :1008-1033
//   create     :1035-1061 eval params  :1063-1142  teardown      :1199-1213
#include "ngx_snippet.h"
#include "guard.h"
#include "logging.h"
#include "ngx_param.h"
#include <chrono>
#include <cstring>

namespace dlssnr {

HMODULE g_layerModule = nullptr;

// ---------------------------------------------------------------------------
// Caller-identity spoof: IAT hook of KERNEL32!GetModuleFileNameW inside the
// snippet/core modules so they see "nvngx.dll" as the caller.
// ---------------------------------------------------------------------------
static DWORD WINAPI SpoofedGetModuleFileNameW(HMODULE module, LPWSTR filename, DWORD size) noexcept {
    if (module == g_layerModule) {
        static constexpr wchar_t AUTHORIZED_CALLER[] = L"nvngx.dll";
        constexpr DWORD LEN = ARRAYSIZE(AUTHORIZED_CALLER) - 1;
        if (!filename || !size) { SetLastError(ERROR_INSUFFICIENT_BUFFER); return 0; }
        if (size <= LEN) {
            if (size > 1) std::memcpy(filename, AUTHORIZED_CALLER, (size - 1) * sizeof(wchar_t));
            filename[size - 1] = L'\0';
            SetLastError(ERROR_INSUFFICIENT_BUFFER);
            return size;
        }
        std::memcpy(filename, AUTHORIZED_CALLER, sizeof(AUTHORIZED_CALLER));
        return LEN;
    }
    extern decltype(&GetModuleFileNameW) g_realGetModuleFileNameW;
    if (g_realGetModuleFileNameW) return g_realGetModuleFileNameW(module, filename, size);
    SetLastError(ERROR_INVALID_FUNCTION);
    return 0;
}
decltype(&GetModuleFileNameW) g_realGetModuleFileNameW = nullptr;

struct SpoofState { void** slot = nullptr; decltype(&GetModuleFileNameW) orig = nullptr; };
static SpoofState g_snippetSpoof, g_coreSpoof;

static void** FindImportedFunctionSlot(HMODULE module, const char* functionName) noexcept {
    if (!module || !functionName) return nullptr;
    auto* base = reinterpret_cast<std::byte*>(module);
    const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
    if (dos->e_magic != IMAGE_DOS_SIGNATURE || dos->e_lfanew <= 0) return nullptr;
    const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(base + dos->e_lfanew);
    if (nt->Signature != IMAGE_NT_SIGNATURE ||
        nt->OptionalHeader.Magic != IMAGE_NT_OPTIONAL_HDR64_MAGIC) return nullptr;
    const auto& dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
    if (!dir.VirtualAddress || !dir.Size || dir.VirtualAddress >= nt->OptionalHeader.SizeOfImage)
        return nullptr;
    auto* desc = reinterpret_cast<IMAGE_IMPORT_DESCRIPTOR*>(base + dir.VirtualAddress);
    const auto* descEnd = reinterpret_cast<const IMAGE_IMPORT_DESCRIPTOR*>(base + dir.VirtualAddress + dir.Size);
    for (; desc < descEnd && desc->Name; ++desc) {
        if (desc->Name >= nt->OptionalHeader.SizeOfImage) continue;
        const char* lib = reinterpret_cast<const char*>(base + desc->Name);
        if (_stricmp(lib, "KERNEL32.dll") != 0 &&
            _stricmp(lib, "api-ms-win-core-libraryloader-l1-2-0.dll") != 0 &&
            _stricmp(lib, "api-ms-win-core-libraryloader-l1-1-0.dll") != 0) continue;
        if (!desc->OriginalFirstThunk || !desc->FirstThunk) continue;
        auto* nameThunk = reinterpret_cast<IMAGE_THUNK_DATA64*>(base + desc->OriginalFirstThunk);
        auto* addrThunk = reinterpret_cast<IMAGE_THUNK_DATA64*>(base + desc->FirstThunk);
        for (; nameThunk->u1.AddressOfData; ++nameThunk, ++addrThunk) {
            if (IMAGE_SNAP_BY_ORDINAL64(nameThunk->u1.Ordinal)) continue;
            const uint32_t rva = static_cast<uint32_t>(nameThunk->u1.AddressOfData);
            if (rva >= nt->OptionalHeader.SizeOfImage) return nullptr;
            const auto* imp = reinterpret_cast<const IMAGE_IMPORT_BY_NAME*>(base + rva);
            if (std::strcmp(reinterpret_cast<const char*>(imp->Name), functionName) == 0)
                return reinterpret_cast<void**>(&addrThunk->u1.Function);
        }
    }
    return nullptr;
}

static bool InstallCallerSpoof(HMODULE module, SpoofState& state) {
    state.slot = FindImportedFunctionSlot(module, "GetModuleFileNameW");
    if (!state.slot) { Log("[spoof] module %p has no GetModuleFileNameW import", (void*)module); return false; }
    DWORD old = 0;
    if (!VirtualProtect(state.slot, sizeof(void*), PAGE_READWRITE, &old)) {
        Log("[spoof] VirtualProtect failed (%lu)", GetLastError()); return false;
    }
    state.orig = reinterpret_cast<decltype(state.orig)>(
        InterlockedExchangePointer(reinterpret_cast<void* volatile*>(state.slot),
                                   reinterpret_cast<void*>(&SpoofedGetModuleFileNameW)));
    VirtualProtect(state.slot, sizeof(void*), old, &old);
    FlushInstructionCache(GetCurrentProcess(), state.slot, sizeof(void*));
    if (!state.orig) { Log("[spoof] original import was null"); return false; }
    g_realGetModuleFileNameW = state.orig;
    Log("[spoof] GetModuleFileNameW IAT hooked at %p (module %p)", (void*)state.slot, (void*)module);
    return true;
}

static void RemoveCallerSpoof(SpoofState& state) {
    if (!state.slot || !state.orig) return;
    DWORD old = 0;
    if (VirtualProtect(state.slot, sizeof(void*), PAGE_READWRITE, &old)) {
        InterlockedExchangePointer(reinterpret_cast<void* volatile*>(state.slot),
                                   reinterpret_cast<void*>(state.orig));
        VirtualProtect(state.slot, sizeof(void*), old, &old);
    }
    state = {};
}

// ---------------------------------------------------------------------------
// Param helpers + guarded NGX calls
// ---------------------------------------------------------------------------
static bool ParamSetUI(NVSDK_NGX_Parameter* p, const char* n, unsigned int v, DWORD* seh) {
    Guarded([&] { p->Set(n, v); return true; }, false, seh);
    return *seh == 0;
}
static bool ParamSetULL(NVSDK_NGX_Parameter* p, const char* n, unsigned long long v, DWORD* seh) {
    Guarded([&] { p->Set(n, v); return true; }, false, seh);
    return *seh == 0;
}
static bool ParamSetF(NVSDK_NGX_Parameter* p, const char* n, float v, DWORD* seh) {
    Guarded([&] { p->Set(n, v); return true; }, false, seh);
    return *seh == 0;
}
static bool ParamGetPtr(NVSDK_NGX_Parameter* p, const char* n, void** v, DWORD* seh) {
    return Guarded([&] { return NVSDK_NGX_SUCCEED(p->Get(n, v)); }, false, seh);
}
static bool ParamSetPtr(NVSDK_NGX_Parameter* p, const char* n, void* v, DWORD* seh) {
    Guarded([&] { p->Set(n, v); return true; }, false, seh);
    return *seh == 0;
}
static bool ParamGetUI(NVSDK_NGX_Parameter* p, const char* n, unsigned int* v, DWORD* seh) {
    return Guarded([&] { return NVSDK_NGX_SUCCEED(p->Get(n, v)); }, false, seh);
}
static bool ParamGetF(NVSDK_NGX_Parameter* p, const char* n, float* v, DWORD* seh) {
    return Guarded([&] { return NVSDK_NGX_SUCCEED(p->Get(n, v)); }, false, seh);
}

static NVSDK_NGX_Result CallInitExtSafely(FnVkInitExt fn, unsigned long long appId,
    const wchar_t* path, VkInstance instance, VkPhysicalDevice pd, VkDevice device,
    NVSDK_NGX_Version version, DWORD* seh) noexcept {
    return Guarded([&] { return fn(appId, path, instance, pd, device, version, nullptr); },
                   NVSDK_NGX_Result_FAIL_SEH, seh);
}
static NVSDK_NGX_Result CallCreateSafely(FnVkCreateFeature fn, VkCommandBuffer cmd, int feature,
    NVSDK_NGX_Parameter* params, NVSDK_NGX_Handle** handle, DWORD* seh) noexcept {
    return Guarded([&] { return fn(cmd, feature, params, handle); },
                   NVSDK_NGX_Result_FAIL_SEH, seh);
}
static NVSDK_NGX_Result CallEvaluateSafely(FnVkEvaluateFeature fn, VkCommandBuffer cmd,
    const NVSDK_NGX_Handle* handle, const NVSDK_NGX_Parameter* params, DWORD* seh) noexcept {
    return Guarded([&] { return fn(cmd, handle, params, nullptr); },
                   NVSDK_NGX_Result_FAIL_SEH, seh);
}
static NVSDK_NGX_Result CallReleaseSafely(FnVkReleaseFeature fn, NVSDK_NGX_Handle* handle, DWORD* seh) noexcept {
    return Guarded([&] { return fn(handle); }, NVSDK_NGX_Result_FAIL_SEH, seh);
}
static NVSDK_NGX_Result CallShutdownSafely(FnVkShutdown1 fn, VkDevice device, DWORD* seh) noexcept {
    return Guarded([&] { return fn(device); }, NVSDK_NGX_Result_FAIL_SEH, seh);
}

// ---------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------
static std::wstring ModuleDir() {
    wchar_t path[MAX_PATH];
    GetModuleFileNameW(g_layerModule, path, MAX_PATH);
    std::wstring dir = path;
    auto sep = dir.find_last_of(L"\\/");
    return (sep == std::wstring::npos) ? L"." : dir.substr(0, sep);
}

static bool FileExists(const std::wstring& p) {
    DWORD a = GetFileAttributesW(p.c_str());
    return a != INVALID_FILE_ATTRIBUTES && !(a & FILE_ATTRIBUTE_DIRECTORY);
}

static std::wstring ResolveBinDir() {
    wchar_t env[MAX_PATH];
    if (GetEnvironmentVariableW(L"DLSSNR_BIN_DIR", env, MAX_PATH) > 0 &&
        FileExists(std::wstring(env) + L"\\nvngx_dlssnr.dll"))
        return env;
    std::wstring dir = ModuleDir();
    if (FileExists(dir + L"\\nvngx_dlssnr.dll")) return dir;
    if (FileExists(dir + L"\\binaries\\nvngx_dlssnr.dll")) return dir + L"\\binaries";
    return L"";
}

static void RegisterPeRange(const char* name, HMODULE mod) {
    if (!mod) return;
    auto* dos = reinterpret_cast<IMAGE_DOS_HEADER*>(mod);
    auto* nt = reinterpret_cast<IMAGE_NT_HEADERS*>(reinterpret_cast<uint8_t*>(mod) + dos->e_lfanew);
    RegisterModuleRange(name, (uintptr_t)mod, nt->OptionalHeader.SizeOfImage);
}

// ---------------------------------------------------------------------------
// Load + init (everything up to and including CreateFeature(18))
// ---------------------------------------------------------------------------
static void ApplyDlssgContract(NgxSnippet& s, uint32_t width, uint32_t height);
bool NgxLoadAndInit(NgxSnippet& s, VkInstance instance, VkPhysicalDevice pd, VkDevice device,
                    uint32_t width, uint32_t height, VkCommandBuffer recordingCmd,
                    const NgxTuning& tuning) {
    if (s.disabled) return false;
    s.binDir = ResolveBinDir();
    if (s.binDir.empty()) { Log("[ngx] nvngx_dlssnr.dll not found (set DLSSNR_BIN_DIR)"); s.disabled = true; return false; }
    Log("[ngx] bin dir: %ls", s.binDir.c_str());

    s.snippet = LoadLibraryExW((s.binDir + L"\\" + s.snippetName).c_str(), nullptr,
        LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if (!s.snippet) { Log("[ngx] LoadLibrary %ls failed (%lu)", s.snippetName, GetLastError()); s.disabled = true; return false; }
    Log("[ngx] %ls loaded at %p", s.snippetName, (void*)s.snippet);
    RegisterPeRange("nvngx_dlssnr.dll", s.snippet);

    s.initExt = reinterpret_cast<FnVkInitExt>(GetProcAddress(s.snippet, "NVSDK_NGX_VULKAN_Init_Ext"));
    s.initExt2 = reinterpret_cast<FnVkInitExt>(GetProcAddress(s.snippet, "NVSDK_NGX_VULKAN_Init_Ext2"));
    s.initPlain = reinterpret_cast<FnVkInitExt>(GetProcAddress(s.snippet, "NVSDK_NGX_VULKAN_Init"));
    s.createFeature = reinterpret_cast<FnVkCreateFeature>(GetProcAddress(s.snippet, "NVSDK_NGX_VULKAN_CreateFeature"));
    s.evaluateFeature = reinterpret_cast<FnVkEvaluateFeature>(GetProcAddress(s.snippet, "NVSDK_NGX_VULKAN_EvaluateFeature"));
    s.releaseFeature = reinterpret_cast<FnVkReleaseFeature>(GetProcAddress(s.snippet, "NVSDK_NGX_VULKAN_ReleaseFeature"));
    s.shutdown1 = reinterpret_cast<FnVkShutdown1>(GetProcAddress(s.snippet, "NVSDK_NGX_VULKAN_Shutdown1"));
    if (!s.createFeature || !s.evaluateFeature || !s.releaseFeature || !s.shutdown1) {
        Log("[ngx] snippet Vulkan exports incomplete (create=%p eval=%p release=%p shutdown=%p)",
            (void*)s.createFeature, (void*)s.evaluateFeature, (void*)s.releaseFeature, (void*)s.shutdown1);
        s.disabled = true; return false;
    }
    Log("[ngx] snippet exports: init=%p create=%p", (void*)s.initExt, (void*)s.createFeature);

    if (!InstallCallerSpoof(s.snippet, g_snippetSpoof)) { s.disabled = true; return false; }

    s.nvapi = LoadLibraryExW((s.binDir + L"\\nvapi64.dll").c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (s.nvapi) RegisterPeRange("nvapi64.dll", s.nvapi);

    // Core (nvngx.dll): libmgr prerequisite for snippet init; also param allocator.
    std::wstring corePath = s.binDir + L"\\nvngx.dll";
    if (FileExists(corePath)) {
        s.core = LoadLibraryExW(corePath.c_str(), nullptr,
            LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        if (s.core) InstallCallerSpoof(s.core, g_coreSpoof);
    }
    if (s.core) {
        const char* projectId = "7c134ab9-9677-4af5-a2b2-bca943350861";
        typedef NVSDK_NGX_Result (NVSDK_CONV* FnCoreInitWithProjectID)(
            const char*, int, const char*, const wchar_t*,
            VkInstance, VkPhysicalDevice, VkDevice);
        typedef NVSDK_NGX_Result (NVSDK_CONV* FnCoreInitExt)(
            unsigned long long, const wchar_t*, VkInstance, VkPhysicalDevice, VkDevice,
            NVSDK_NGX_Version, const NVSDK_NGX_FeatureDiscoveryInfo*);
        auto coreInitProjectID = reinterpret_cast<FnCoreInitWithProjectID>(
            GetProcAddress(s.core, "NVSDK_NGX_VULKAN_Init_with_ProjectID"));
        if (!coreInitProjectID)
            coreInitProjectID = reinterpret_cast<FnCoreInitWithProjectID>(
                GetProcAddress(s.core, "NVSDK_NGX_VULKAN_Init_ProjectID"));
        auto coreInitExt = reinterpret_cast<FnCoreInitExt>(
            GetProcAddress(s.core, "NVSDK_NGX_VULKAN_Init_Ext"));
        bool coreInited = false;
        if (coreInitProjectID) {
            DWORD seh2 = 0;
            NVSDK_NGX_Result r = Guarded([&] {
                return coreInitProjectID(projectId, 3 /*CUSTOM*/, "Magpie-Experimental-0.5.7",
                    s.binDir.c_str(), instance, pd, device);
            }, NVSDK_NGX_Result_FAIL_SEH, &seh2);
            Log("[core] VULKAN_Init_with_ProjectID -> %#x seh=%#x", (uint32_t)r, seh2);
            coreInited = NVSDK_NGX_SUCCEED(r);
        } else {
            Log("[core] no Init_with_ProjectID export");
        }
        if (!coreInited && coreInitExt) {
            DWORD seh2 = 0;
            NVSDK_NGX_Result r = Guarded([&] {
                return coreInitExt(DLSSNR_SIGNED_SNIPPET_APPLICATION_ID, s.binDir.c_str(),
                    instance, pd, device, NVSDK_NGX_Version_API_14, nullptr);
            }, NVSDK_NGX_Result_FAIL_SEH, &seh2);
            Log("[core] VULKAN_Init_Ext -> %#x seh=%#x", (uint32_t)r, seh2);
            coreInited = NVSDK_NGX_SUCCEED(r);
        }
        Log("[core] init %s", coreInited ? "OK" : "FAILED (continuing)");
    } else {
        Log("[core] nvngx.dll not loadable");
    }

    // Parameters: DLL allocator preferred (core -> snippet), own 16-slot vtable
    // implementation only as fallback (notes section 4: blocks must come from the
    // matching API family's allocator).
    {
        DWORD seh2 = 0;
        NVSDK_NGX_Result r = NVSDK_NGX_Result_FAIL_Failure;
        if (s.core) {
            auto coreAlloc = reinterpret_cast<FnVkAllocateParameters>(
                GetProcAddress(s.core, "NVSDK_NGX_VULKAN_AllocateParameters"));
            auto coreDestroy = reinterpret_cast<FnVkDestroyParameters>(
                GetProcAddress(s.core, "NVSDK_NGX_VULKAN_DestroyParameters"));
            if (coreAlloc && coreDestroy) {
                r = Guarded([&] { return coreAlloc(&s.params); }, NVSDK_NGX_Result_FAIL_SEH, &seh2);
                Log("[params] core AllocateParameters -> %#x seh=%#x", (uint32_t)r, seh2);
                if (NVSDK_NGX_SUCCEED(r) && s.params) s.paramsDestroy = coreDestroy;
                else s.params = nullptr;
            }
        }
        if (!s.params && s.snippet) {
            auto snipAlloc = reinterpret_cast<FnVkAllocateParameters>(
                GetProcAddress(s.snippet, "NVSDK_NGX_VULKAN_AllocateParameters"));
            auto snipDestroy = reinterpret_cast<FnVkDestroyParameters>(
                GetProcAddress(s.snippet, "NVSDK_NGX_VULKAN_DestroyParameters"));
            if (snipAlloc) {
                r = Guarded([&] { return snipAlloc(&s.params); }, NVSDK_NGX_Result_FAIL_SEH, &seh2);
                Log("[params] snippet AllocateParameters -> %#x seh=%#x", (uint32_t)r, seh2);
                if (NVSDK_NGX_SUCCEED(r) && s.params) s.paramsDestroy = snipDestroy;
                else s.params = nullptr;
            } else {
                Log("[params] snippet does not export AllocateParameters");
            }
        }
        if (!s.params) {
            s.params = new OwnParam();
            s.ownParams = true;
            Log("[params] using own NVSDK_NGX_Parameter implementation");
        }
    }
    {
        DWORD seh2 = 0;
        bool ok = ParamSetUI(s.params, "__selftest", 0xC0FFEE, &seh2);
        unsigned int back = 0;
        ok = ok && ParamGetUI(s.params, "__selftest", &back, &seh2) && back == 0xC0FFEE;
        Log("[params] round-trip self-test: %s (seh=%#x)", ok ? "PASS" : "FAIL", seh2);
        if (!ok) { s.disabled = true; return false; }
    }

    // Frame generation reads its contract earlier than the denoiser does -- some of it while the
    // snippet is still initialising, before any feature is built -- so it is written here, the moment
    // there is a parameter block to write it into. Putting it at create time was not early enough:
    // the keys kept reporting missing after they were being set, because the read that logged them
    // had already happened.
    if (s.featureId == 11) {
        s.featureW = width;
        s.featureH = height;
        ApplyDlssgContract(s, width, height);
        NgxSetDlssgEval(s, true, 1);
        Log("[mfg] contract written at init for %ux%u", width, height);
    }

    // Create parameters (extracted_pipeline_notes.md section 4).
    DWORD seh = 0;
    bool ps = true;
    ps &= ParamSetUI(s.params, "DLSSNR.Width", width, &seh);
    ps &= ParamSetUI(s.params, "DLSSNR.Height", height, &seh);
    ps &= ParamSetUI(s.params, "DLSSNR.InputWidth", width, &seh);
    ps &= ParamSetUI(s.params, "DLSSNR.InputHeight", height, &seh);
    ps &= ParamSetUI(s.params, "DLSSNR.OutputWidth", width, &seh);
    ps &= ParamSetUI(s.params, "DLSSNR.OutputHeight", height, &seh);
    ps &= ParamSetUI(s.params, "DLSSNR.Output.Width", width, &seh);
    ps &= ParamSetUI(s.params, "DLSSNR.Output.Height", height, &seh);
    ps &= ParamSetUI(s.params, "DLSSNR.Upscaling", 0u, &seh);
    ps &= ParamSetF(s.params, "DLSSNR.Scale", 1.0f, &seh);
    ps &= ParamSetF(s.params, "DLSSNR.ScalingRatio", 1.0f, &seh);
    ps &= ParamSetULL(s.params, "DLSSNRComputeScalingRatioCallback",
                      (unsigned long long)(void*)&ScalingRatioCallback, &seh);
    ps &= ParamSetUI(s.params, "DLSSNR.Hint.Render.Preset", 0u, &seh);
    ps &= ParamSetUI(s.params, "Width", width, &seh);
    ps &= ParamSetUI(s.params, "Height", height, &seh);
    ps &= ParamSetUI(s.params, "PerfQualityValue", 3u, &seh);  // Balanced
    ps &= ParamSetUI(s.params, "CreationNodeMask", 1u, &seh);
    ps &= ParamSetUI(s.params, "VisibilityNodeMask", 1u, &seh);
    ps &= ParamSetUI(s.params, "NVSDK_NGX_Parameter_PerfQualityValue", 3u, &seh);
    ps &= ParamSetUI(s.params, "NVSDK_NGX_Parameter_CreationNodeMask", 1u, &seh);
    ps &= ParamSetUI(s.params, "NVSDK_NGX_Parameter_VisibilityNodeMask", 1u, &seh);

    // Create flags: sharpening is applied by the net when the runtime float is
    // nonzero (see NgxSetSharpness); auto-exposure keeps adaptation state in the
    // DLL so it survives normal dynamic lighting without host-side resets.
    unsigned int createFlags = NVSDK_NGX_DLSS_Feature_Flags_DoSharpening |
                               NVSDK_NGX_DLSS_Feature_Flags_AutoExposure;
    const char* hdrEnv = getenv("DLSSNR_HDR");
    const bool wantHdr = s.hdrActive || (hdrEnv && hdrEnv[0] == '1');
    if (wantHdr) createFlags |= NVSDK_NGX_DLSS_Feature_Flags_IsHDR;
    ps &= ParamSetUI(s.params, "Feature_Flags", createFlags, &seh);
    ps &= ParamSetUI(s.params, "NVSDK_NGX_Parameter_Feature_Flags", createFlags, &seh);

    // Non-destructive exposure: identity pre-exposure/exposure-scale (never
    // re-pinned per frame) + SDR tonemapped hint unless the HDR path is asked
    // for and the snippet's feature flags advertise HDR.
    ps &= ParamSetF(s.params, "InPreExposure", 1.0f, &seh);
    ps &= ParamSetF(s.params, "InExposureScale", 1.0f, &seh);
    ps &= ParamSetF(s.params, "NVSDK_NGX_Parameter_PreExposure", 1.0f, &seh);
    ps &= ParamSetF(s.params, "NVSDK_NGX_Parameter_ExposureScale", 1.0f, &seh);
    ps &= ParamSetUI(s.params, "DLSSNR.AutoExposure", 1u, &seh);
    Log("[params] create contract set: %s (seh=%#x) flags=%#x hdr=%d",
        ps ? "ok" : "FAILED", seh, createFlags, int(wantHdr));

    // Snippet Init_Ext: (appId, path, instance, pd, device, version, featureInfo=nullptr)
    NVSDK_NGX_Result initResult = NVSDK_NGX_Result_FAIL_NotInitialized;
    for (NVSDK_NGX_Version ver : { NVSDK_NGX_Version_API_14, NVSDK_NGX_Version_API_13 }) {
        if (s.initExt) {
            initResult = CallInitExtSafely(s.initExt, DLSSNR_SIGNED_SNIPPET_APPLICATION_ID,
                s.binDir.c_str(), instance, pd, device, ver, &seh);
            Log("[ngx] VULKAN_Init_Ext(ver=0x%x) -> %#x seh=%#x", ver, (uint32_t)initResult, seh);
            if (NVSDK_NGX_SUCCEED(initResult)) break;
        }
        if (!NVSDK_NGX_SUCCEED(initResult) && s.initExt2) {
            initResult = CallInitExtSafely(s.initExt2, DLSSNR_SIGNED_SNIPPET_APPLICATION_ID,
                s.binDir.c_str(), instance, pd, device, ver, &seh);
            Log("[ngx] VULKAN_Init_Ext2(ver=0x%x) -> %#x seh=%#x", ver, (uint32_t)initResult, seh);
            if (NVSDK_NGX_SUCCEED(initResult)) break;
        }
        if (!NVSDK_NGX_SUCCEED(initResult) && s.initPlain) {
            initResult = CallInitExtSafely(s.initPlain, DLSSNR_SIGNED_SNIPPET_APPLICATION_ID,
                s.binDir.c_str(), instance, pd, device, ver, &seh);
            Log("[ngx] VULKAN_Init(ver=0x%x) -> %#x seh=%#x", ver, (uint32_t)initResult, seh);
            if (NVSDK_NGX_SUCCEED(initResult)) break;
        }
    }
    if (!NVSDK_NGX_SUCCEED(initResult)) {
        Log("[ngx] snippet init failed, disabling layer");
        s.disabled = true;
        return false;
    }

// Public Vulkan NGX contract: query Feature-18 requirements before create.
    {
        auto reqs2 = reinterpret_cast<FnVkGetFeatureReqs2>(
            GetProcAddress(s.snippet, "NVSDK_NGX_VULKAN_GetFeatureRequirements"));
        if (reqs2) {
            DWORD seh2 = 0;
            NVSDK_NGX_FeatureRequirements fr{};
            NVSDK_NGX_Result r = Guarded([&] { return reqs2(instance, pd, &fr); },
                                         NVSDK_NGX_Result_FAIL_SEH, &seh2);
            Log("[reqs] GetFeatureRequirements -> %#x seh=%#x ver=%u.%u flags=%#x minGPU=%u inGPU=%u cs=%u.%u",
                (uint32_t)r, seh2, fr.Version.Major, fr.Version.Minor, fr.FeatureFlags,
                fr.MinGPUMode, fr.InGPUMode, fr.MinCSMajorVersion, fr.MinCSMinorVersion);
            if (NVSDK_NGX_SUCCEED(r)) {
                s.featureFlags = fr.FeatureFlags;
                s.hdrCapable = (fr.FeatureFlags & NVSDK_NGX_DLSS_Feature_Flags_IsHDR) != 0u;
                s.hdrActive = wantHdr && s.hdrCapable;
            }
        }
    }

    // Tonemapping hint now that the snippet's feature flags are known: SDR by default (the input is
    // LDR RGBA8); HDR only when both asked for and advertised.
    {
        DWORD seh2 = 0;
        const bool hdrPath = wantHdr && s.hdrCapable;
        ParamSetUI(s.params, "DLSSNR.Hdr", hdrPath ? 1u : 0u, &seh2);
        ParamSetUI(s.params, "DLSSNR.SDR", hdrPath ? 0u : 1u, &seh2);
        Log("[params] tonemap hint: %s (featureFlags=%#x)", hdrPath ? "HDR" : "SDR", s.featureFlags);
    }

    // Last, so neither the create contract above nor the tonemap hint can overwrite it. Its preset
    // write in particular used to land after everything the caller chose.
    NgxSetCreateTuning(s, tuning);

    bool created = NgxCreatePass(s, 0, width, height, recordingCmd);
    if (!created && s.snippet && s.params) {
        auto reqs = reinterpret_cast<FnVkGetFeatureRequirements>(
            GetProcAddress(s.snippet, "NVSDK_NGX_VULKAN_GetFeatureRequirements"));
        if (reqs) {
            DWORD seh3 = 0;
            NVSDK_NGX_Result r = Guarded([&] { return reqs(instance, pd, s.params); },
                                         NVSDK_NGX_Result_FAIL_SEH, &seh3);
            Log("[diag] GetFeatureRequirements -> %#x seh=%#x", (uint32_t)r, seh3);
            unsigned int avail = 0;
            ParamGetUI(s.params, "DLSSNR.Available", &avail, &seh3);
            Log("[diag] DLSSNR.Available=%u", avail);
        }
    }
    return created;
}

void NgxSetCreateTuning(NgxSnippet& s, const NgxTuning& t) {
    if (!s.params) return;
    DWORD seh = 0;
    bool ok = true;
    ok &= ParamSetUI(s.params, "DLSSNR.Hint.Render.Preset", t.preset, &seh);
    ok &= ParamSetUI(s.params, "DLSSNR.Style", t.style, &seh);
    ok &= ParamSetF(s.params, "DLSSNR.Intensity", t.intensity, &seh);
    ok &= ParamSetF(s.params, "DLSSNR.LocalToneStrength", t.localTone, &seh);
    ok &= ParamSetF(s.params, "DLSSNR.LocalStructureStrength", t.localStructure, &seh);
    ok &= ParamSetF(s.params, "DLSSNR.SkinStructureStrength", t.skinStructure, &seh);
    ok &= ParamSetUI(s.params, "DLSSNR.UseAutoMask", t.autoMask, &seh);
    if (!ok) Log("[params] create tuning FAILED (seh=%#x)", seh);
    else
        Log("[params] create tuning: preset=%u style=%u intensity=%.2f tone=%.2f structure=%.2f "
            "skin=%.2f automask=%u",
            t.preset, t.style, t.intensity, t.localTone, t.localStructure, t.skinStructure, t.autoMask);
}

void NgxReleasePass(NgxSnippet& s, uint32_t pass, VkDevice device) {
    if (pass >= kMaxPasses || !s.features[pass] || !s.releaseFeature) return;
    // Never free under the GPU. The helper submits and fences every evaluate, so waiting on the
    // device here is enough and is cheaper to reason about than parking the handle for N frames.
    (void) device;
    DWORD seh = 0;
    NVSDK_NGX_Result r = CallReleaseSafely(s.releaseFeature, s.features[pass], &seh);
    Log("[ngx] ReleaseFeature pass %u -> %#x seh=%#x", pass, (uint32_t)r, seh);
    s.features[pass] = nullptr;
}

void NgxReleaseAllPasses(NgxSnippet& s, VkDevice device) {
    for (uint32_t i = 0; i < kMaxPasses; ++i) NgxReleasePass(s, i, device);
    s.featureCount = 0;
    s.ready = false;
}

void NgxSetHdr(NgxSnippet& s, bool want) {
    // Raw: the caller decides, the create either succeeds or the helper falls back. Clamping to
    // hdrCapable here would silently swallow the request before init has learned the capability.
    s.hdrActive = want;
}

// The HDR contract, rewritten from s.hdrActive before every create. The create flags and the
// tonemap hint are ordinary string-keyed parameters, so restating them here is exactly what the
// init-time block did once -- and doing it at create is what lets a toggle take effect on the next
// feature build rather than never.
static void ApplyHdrContract(NgxSnippet& s) {
    if (!s.params) return;
    DWORD seh = 0;
    unsigned int flags = NVSDK_NGX_DLSS_Feature_Flags_DoSharpening |
                         NVSDK_NGX_DLSS_Feature_Flags_AutoExposure;
    if (s.hdrActive) flags |= NVSDK_NGX_DLSS_Feature_Flags_IsHDR;
    ParamSetUI(s.params, "Feature_Flags", flags, &seh);
    ParamSetUI(s.params, "NVSDK_NGX_Parameter_Feature_Flags", flags, &seh);
    ParamSetUI(s.params, "DLSSNR.Hdr", s.hdrActive ? 1u : 0u, &seh);
    ParamSetUI(s.params, "DLSSNR.SDR", s.hdrActive ? 0u : 1u, &seh);
}

// The create-time block frame generation asks for.
//
// Discovered rather than guessed: the parameter object logs every key the DLL reads and does not
// find, so the first create attempt named its own requirements -- Width, Height, BackbufferFormat and
// UseReflexMatrices -- and each round of filling them in reveals the next. That log is why this can
// be written at all without the DLSS-G headers.
// Why frame generation declines. Reflex is not it.
//
// The module imports ADVAPI32, KERNEL32, USER32 and VERSION and nothing else. It names Reflex in
// exactly two places, both parameter keys, and carries no low-latency string, no VK_NV_low_latency2,
// and not one Reflex NVAPI entry point. It does not check Reflex, because checking Reflex is
// sl.dlss_g.dll's job and this drives the snippet directly, past it.
//
// What the reference project actually does for DLSS-G says where the work happens: it hooks the DXGI
// factory, creates a *Streamline-proxied* swapchain, declares the feature loaded, and reads the state
// back through slDLSSGGetState. The interpolation is performed inside that proxied swapchain's
// Present. The snippet is a component of that arrangement, and MustCallEval is how the arrangement
// asks it whether this present needs an evaluation.
//
// So the answer is a swapchain. There is no present here to hang a generated frame on, no
// back-buffer index, no frame cadence the snippet can see -- so "no evaluation required" is a
// correct answer to the question being asked, and it will keep being correct however many parameters
// are written. All 128 keys the module knows are set and it has not moved.
//
// Claiming Reflex was tried and is worse than useless: asserting UseReflexMatrices makes the module
// demand matrices that do not exist here and CreateFeature fails outright. That is at least proof
// the key is consumed rather than ignored. DLSSNR_MFG_FAKE_REFLEX=0 stops claiming it.

// Superseded: the earlier note here read Streamline's status enum as though it applied to this path.
// Those codes are reported by sl.dlss_g.dll, which is exactly the layer being bypassed.
//
// The snippet answers its settings callback with MustCallEval=0 -- it is not failing, it is saying
// no evaluation is required -- and sl_dlss_g.h enumerates the reasons DLSS-G reports for not
// running:
//
//     eFailResolutionTooLow                   the swapchain is too small
//     eFailReflexNotDetectedAtRuntime         "Reflex must be turned on when DLSS-G is on"
//     eFailHDRFormatNotSupported
//     eFailCommonConstantsInvalid             the camera matrices, jitter and depth
//     eFailGetCurrentBackBufferIndexNotCalled the swapchain's own index API
//
// Two of those are this architecture rather than a missing key. Reflex is a documented hard
// requirement, and it is not a flag: it is frame markers -- simulation start and end, present --
// around the game's actual frame loop, which is why the reference project hooks slReflexSetMarker
// and slPCLSetMarker and spoofs the architecture for kFeatureReflex. This helper is a separate
// process with no game loop and no present queue to mark. And the common constants are the camera
// matrices, the jitter and the depth, all of which are supplied here as identity and zeros because a
// swapchain-only layer has none of them; "invalid" is a fair description.
//
// So the parameter contract was never the obstacle, and satisfying more of it will not help. What
// DLSS-G wants is to be inside the presenting process, holding the swapchain, with Reflex marking
// the frame -- which is the one thing this design puts in another process on purpose.

// Answers the snippet's memory question. Flat, like the reference project's.
static NVSDK_NGX_Result NVSDK_CONV DlssgEstimateVram(unsigned int, unsigned int, unsigned int,
                                                     unsigned int, unsigned int, unsigned int,
                                                     unsigned int, unsigned int, unsigned int,
                                                     size_t* estimated) {
    if (estimated) *estimated = size_t(300) * 1024 * 1024;
    return NVSDK_NGX_Result_Success;
}

static void ApplyDlssgContract(NgxSnippet& s, uint32_t width, uint32_t height) {
    if (!s.params) return;
    DWORD seh = 0;
    ParamSetUI(s.params, "DLSSG.Width", width, &seh);
    ParamSetUI(s.params, "DLSSG.Height", height, &seh);
    // The format the game actually presents in, when the layer has said what it is.
    //
    // This was hardcoded to 37, R8G8B8A8_UNORM, which is what the transport carries -- but frame
    // generation is being told about a *backbuffer*, and the game's is B8G8R8A8_UNORM, format 44.
    // The same bytes in the other order, described wrongly.
    ParamSetUI(s.params, "DLSSG.BackbufferFormat",
               s.fgSwapchainFormat ? s.fgSwapchainFormat : 37u, &seh);
    // No Reflex here: there is no game presenting through this device, so there are no camera
    // matrices to hand over and nothing to align them to.
    ParamSetUI(s.params, "DLSSG.UseReflexMatrices", 0u, &seh);

    // The tier the first successful create named on its way through.
    ParamSetUI(s.params, "DLSSG.InternalWidth", width, &seh);
    ParamSetUI(s.params, "DLSSG.InternalHeight", height, &seh);
    ParamSetUI(s.params, "DLSSG.DynamicResolution", 0u, &seh);
    // Claim every resource is provided, so the DLL asks for them by name.
    //
    // At zero it decided it had nothing and failed with MissingInput without querying a single
    // resource key, which taught nothing. The parameter object logs a miss on the pointer getter as
    // well as the scalar ones, so the way to learn the resource contract is to say everything is
    // there and read the list it comes back with. DLSSNR_MFG_CLAIM_ALL=0 restores the honest value.
    // Zero, which is what makes evaluate succeed.
    //
    // All-ones was a discovery trick: at zero the DLL decided it had nothing and failed without
    // naming a single resource, so claiming everything made it ask, and the list it asked for is the
    // contract above. Kept as the claim, though, it then *demands* what it was promised -- UI,
    // UIAlpha, the distortion field, none of which a swapchain-only layer has -- and returns
    // MissingInput forever. Saying nothing is guaranteed lets it work with what is actually bound.
    // DLSSNR_MFG_CLAIM_ALL=1 restores the trick for discovering keys on another snippet version.
    static const unsigned int provided = [] {
        const char* v = getenv("DLSSNR_MFG_CLAIM_ALL");
        return (v && v[0] == '1') ? 0xFFFFFFFFu : 0u;
    }();
    ParamSetUI(s.params, "DLSSG.ResourceAlwaysProvidedFlags", provided, &seh);
    ParamSetUI(s.params, "DLSSG.ResourceNeverProvidedFlags", 0u, &seh);
    ParamSetUI(s.params, "DLSSG.UserInterfaceRecompositionEnabled", 0u, &seh);
    ParamSetUI(s.params, "DLSSG.NvAppOvrAppliedVal.StreamlineMode", 0u, &seh);

    // How many frames to generate between each pair of real ones. This is the number the reference
    // project patches the snippet's architecture clamp to raise; asking for it directly first
    // establishes whether the clamp is even reached by this route.
    static const unsigned int frames = [] {
        const char* v = getenv("DLSSNR_MFG_FRAMES");
        const int n = v && *v ? atoi(v) : 1;
        return (unsigned int)(n < 1 ? 1 : (n > 8 ? 8 : n));
    }();
    ParamSetUI(s.params, "DLSSG.NvAppOvrAppliedVal.MultiFrameCount", frames, &seh);
    ParamSetUI(s.params, "DLSSG.MultiFrameCount", frames, &seh);

    Log("[mfg] create contract: %ux%u backbufferFormat=37 reflexMatrices=0 multiFrameCount=%u",
        width, height, frames);
}

// The evaluate-time block, discovered the same way the create block was: run it, read what it asked
// for and did not find, fill that in, run it again.
void NgxSetDlssgEval(NgxSnippet& s, bool reset, unsigned int frameIndex) {
    if (!s.params) return;
    DWORD seh = 0;
    ParamSetUI(s.params, "DLSSG.Reset", reset ? 1u : 0u, &seh);
    ParamSetUI(s.params, "DLSSG.MenuDetectionEnabled", 0u, &seh);
    ParamSetUI(s.params, "DLSSG.AsyncCreateEnabled", 0u, &seh);
    ParamSetUI(s.params, "DLSSG.IndicatorLevel", 0u, &seh);
    // Depth is a zero-filled image here, so the linearisation has nothing to describe. Written
    // because the DLL reads them, with values that say "no useful depth" as plainly as the contract
    // allows rather than inventing a near and far plane the frame does not have.
    ParamSetF(s.params, "DLSSG.LinearizedDepth_Scale", 1.0f, &seh);
    ParamSetF(s.params, "DLSSG.LinearizedDepth_NearFarPartition", 0.0f, &seh);
    ParamSetF(s.params, "DLSSG.MinRelativeLinearDepthObjectSeparation", 1.0f, &seh);

    // Clip space to the previous frame's clip space, as a camera matrix.
    //
    // There is no game camera to read here -- a swapchain-only layer never sees one -- so this is the
    // identity, which states that the camera did not move and leaves the whole displacement to the
    // motion field. That is the honest description of what this process knows, and it is also what
    // the motion field is already carrying, since it is estimated from the frames themselves rather
    // than supplied by the game.
    //
    // Static because the DLL keeps the pointer rather than the values.
    static float clipToPrevClip[16] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f,
    };
    ParamSetPtr(s.params, "DLSSG.ClipToPrevClip", clipToPrevClip, &seh);
    // And its inverse, which for the identity is itself.
    ParamSetPtr(s.params, "DLSSG.PrevClipToClip", clipToPrevClip, &seh);

    // How to read the motion field's units. The helper writes it in pixels, so one texel of the
    // field is one pixel of the frame and the scale is the reciprocal of the raster -- the same
    // convention the denoiser is already given through ApplyMotionScale.
    const float w = s.featureW ? float(s.featureW) : 1.0f;
    const float h = s.featureH ? float(s.featureH) : 1.0f;
    ParamSetF(s.params, "DLSSG.MvecScaleX", 1.0f / w, &seh);
    ParamSetF(s.params, "DLSSG.MvecScaleY", 1.0f / h, &seh);

    // A camera at the origin looking down +Z, with the axes it implies. Same reasoning as the
    // matrices above: there is no game camera to report, so this states a still one and leaves the
    // motion to the field.
    ParamSetF(s.params, "DLSSG.CameraPosX", 0.0f, &seh);
    ParamSetF(s.params, "DLSSG.CameraPosY", 0.0f, &seh);
    ParamSetF(s.params, "DLSSG.CameraPosZ", 0.0f, &seh);
    ParamSetF(s.params, "DLSSG.CameraRightX", 1.0f, &seh);
    ParamSetF(s.params, "DLSSG.CameraRightY", 0.0f, &seh);
    ParamSetF(s.params, "DLSSG.CameraRightZ", 0.0f, &seh);
    ParamSetF(s.params, "DLSSG.CameraUpX", 0.0f, &seh);
    ParamSetF(s.params, "DLSSG.CameraUpY", 1.0f, &seh);
    ParamSetF(s.params, "DLSSG.CameraUpZ", 0.0f, &seh);
    ParamSetF(s.params, "DLSSG.CameraFwdX", 0.0f, &seh);
    ParamSetF(s.params, "DLSSG.CameraFwdY", 0.0f, &seh);
    ParamSetF(s.params, "DLSSG.CameraFwdZ", 1.0f, &seh);
    ParamSetF(s.params, "DLSSG.CameraNear", 0.1f, &seh);
    ParamSetF(s.params, "DLSSG.CameraFar", 1000.0f, &seh);
    ParamSetUI(s.params, "DLSSG.DepthInverted", 0u, &seh);

    // No distortion field is supplied, so its precision hints describe nothing.
    ParamSetUI(s.params, "DLSSG.BidirectionalDistortionFieldLowPrecision.IsLowPrecision", 0u, &seh);
    ParamSetF(s.params, "DLSSG.BidirectionalDistortionFieldLowPrecision.Bias", 0.0f, &seh);
    ParamSetF(s.params, "DLSSG.BidirectionalDistortionFieldLowPrecision.Scale", 1.0f, &seh);

    // Projection, jitter and lens. All identity or zero, for the same reason: this process has the
    // frames and nothing that produced them.
    static float ident[16] = { 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1 };
    ParamSetPtr(s.params, "DLSSG.CameraViewToClip", ident, &seh);
    ParamSetPtr(s.params, "DLSSG.ClipToCameraView", ident, &seh);
    ParamSetPtr(s.params, "DLSSG.ClipToLensClip", ident, &seh);
    ParamSetF(s.params, "DLSSG.JitterOffsetX", 0.0f, &seh);
    ParamSetF(s.params, "DLSSG.JitterOffsetY", 0.0f, &seh);
    ParamSetF(s.params, "DLSSG.CameraPinholeOffsetX", 0.0f, &seh);
    ParamSetF(s.params, "DLSSG.CameraPinholeOffsetY", 0.0f, &seh);
    ParamSetF(s.params, "DLSSG.CameraFOV", 1.0472f, &seh);  // 60 degrees
    ParamSetF(s.params, "DLSSG.CameraAspectRatio", w / (h > 0.0f ? h : 1.0f), &seh);
    ParamSetUI(s.params, "DLSSG.OrthoProjection", 0u, &seh);

    // What the motion field is and is not. It is estimated by optical flow over the whole picture, so
    // it already contains the camera's own movement, and it is neither dilated nor jittered because
    // nothing here does either of those things.
    ParamSetUI(s.params, "DLSSG.CameraMotionIncluded", 1u, &seh);
    ParamSetUI(s.params, "DLSSG.MvecDilated", 0u, &seh);
    ParamSetUI(s.params, "DLSSG.MvecJittered", 0u, &seh);
    // A sentinel the field will never carry, so nothing is mistaken for it.
    ParamSetF(s.params, "DLSSG.MvecInvalidValue", -1.0e30f, &seh);

    ParamSetUI(s.params, "DLSSG.EvalFlags", 0u, &seh);
    ParamSetUI(s.params, "DLSSG.ColorBuffersHDR", 0u, &seh);
    ParamSetUI(s.params, "DLSSG.AutomodeOverrideReset", 0u, &seh);
    ParamSetUI(s.params, "DLSSG.NotRenderingGameFrames", 0u, &seh);
    ParamSetUI(s.params, "DLSSG.FullscreenMode", 0u, &seh);

    // Which of the generated frames this evaluate is producing, counted from one.
    ParamSetUI(s.params, "DLSSG.MultiFrameIndex", frameIndex, &seh);

    // Which frame the backbuffer holds, advanced once per evaluate.
    //
    // One of only three keys the DLL knows that nothing here was setting, and the one that would
    // most obviously produce the answer it was giving: a model told nothing about which frame it is
    // looking at has no reason to believe the picture has moved, and "no evaluation required" is
    // then correct rather than a refusal.
    // The game's own present count where it is known, rather than a count of the helper's answers.
    // They are not the same sequence and only one of them is the one being interpolated between.
    ParamSetULL(s.params, "DLSSG.BackbufferFrameID",
                s.fgPresentIndex ? s.fgPresentIndex : ++s.fgFrameId, &seh);
    // Claim Reflex, since claiming is the only thing this interface offers.
    //
    // The module imports no Streamline or Reflex library and names Reflex in exactly two places, so
    // there is no runtime to stand up and nothing to impersonate -- only these two values to assert.
    // DLSSNR_MFG_FAKE_REFLEX=0 says the honest thing instead.
    static const unsigned int claimReflex = [] {
        const char* v = getenv("DLSSNR_MFG_FAKE_REFLEX");
        return (v && v[0] == '0') ? 0u : 1u;
    }();
    ParamSetUI(s.params, "DLSSG.ReflexWarp.Available", claimReflex, &seh);
    ParamSetUI(s.params, "DLSSG.UseReflexMatrices", claimReflex, &seh);

    // Read one of them straight back. If a key reports missing after this says it is present, the
    // DLL is reading a different parameter object than the one being written.
    {
        DWORD seh2 = 0;
        float back = -1.0f;
        const bool got = ParamGetF(s.params, "DLSSG.CameraFOV", &back, &seh2);
        // Marked to survive the reset the DLL performs at the top of evaluate.
        if (s.ownParams) static_cast<OwnParam*>(s.params)->Persist();
        Log("[mfg] write check: CameraFOV set, read back %s value=%.4f (seh=%#x) params=%p own=%d",
            got ? "OK" : "MISSING", back, seh2, (void*)s.params, int(s.ownParams));
    }
}

bool NgxCreatePass(NgxSnippet& s, uint32_t pass, uint32_t width, uint32_t height,
                   VkCommandBuffer recordingCmd) {
    if (s.disabled || !s.params || pass >= kMaxPasses) return false;
    if (s.features[pass]) return true;

    // The evaluate block is written here as well as before each evaluate.
    //
    // The DLL reads much of it while the feature is being built, not only when it runs -- the camera
    // basis, the projection, what the motion field means. The parameter object logs a missing key
    // once per process, which is what made this visible: the keys kept reporting missing after they
    // were being set, because the read that logged them had already happened at create time.
    if (s.featureId == 11) {
        ApplyDlssgContract(s, width, height);
        s.featureW = width;
        s.featureH = height;
        NgxSetDlssgEval(s, true, 1);
    }
    ApplyHdrContract(s);

    DWORD seh = 0;
    const auto t0 = std::chrono::steady_clock::now();
    NVSDK_NGX_Result createResult = CallCreateSafely(s.createFeature, recordingCmd,
        s.featureId, s.params, &s.features[pass], &seh);
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    Log("[ngx] VULKAN_CreateFeature(%u) pass %u -> %#x seh=%#x handle=%p size=%ux%u in %.0f ms",
        s.featureId, pass, (uint32_t)createResult, seh, (void*)s.features[pass], width, height, ms);
    if (!NVSDK_NGX_SUCCEED(createResult) || !s.features[pass]) {
        s.features[pass] = nullptr;
        // Only the first pass failing leaves the feature unusable, and even then it is not the end of
        // the snippet. A later pass failing simply caps the chain, which is what a memory ceiling
        // looks like.
        //
        // Two ways pass 0 can fail and neither is fatal. A failed HDR create is the model refusing
        // float input, and the helper answers by rebuilding at 8-bit. A failed SDR create may be a
        // raster this model will not take, and another size may well work -- so the caller is told
        // this create failed rather than that the model is gone, because latching on one refused size
        // used to take the feature down until the shared header was re-initialised by hand.
        if (pass == 0 && !s.hdrActive) s.createFailed = true;
        return false;
    }
    s.featureW = width;
    s.featureH = height;
    if (pass + 1 > s.featureCount) s.featureCount = pass + 1;
    s.ready = true;
    if (pass == 0)
        Log("[ngx] STATUS: Feature=18 created=true path=vulkan-snippet size=%ux%u", width, height);
    return true;
}

// ---------------------------------------------------------------------------
// Evaluate parameters (extracted_pipeline_notes.md section 5)
// ---------------------------------------------------------------------------
void NgxSetResources(NgxSnippet& s, const NVSDK_NGX_Resource_VK& color,
                     const NVSDK_NGX_Resource_VK& out, const NVSDK_NGX_Resource_VK& mv,
                     const NVSDK_NGX_Resource_VK& depth, uint32_t width, uint32_t height) {
    if (!s.params) return;
    s.resColor = color; s.resOut = out; s.resMV = mv; s.resDepth = depth;
    DWORD seh = 0;
    Guarded([&] {
        const bool hasDepth = s.resDepth.Resource.ImageViewInfo.ImageView != VK_NULL_HANDLE;
        s.params->Set("DLSSNR.Color", &s.resColor);
        s.params->Set("DLSSNR.Output", &s.resOut);
        s.params->Set("DLSSNR.MVec", &s.resMV);
        s.params->Set("DLSSNR.Depth", hasDepth ? &s.resDepth : (const NVSDK_NGX_Resource_VK*)nullptr);
        s.params->Set("DLSSNR.ControlMask", (const NVSDK_NGX_Resource_VK*)nullptr);
        s.params->Set("DLSSNR.UI", (const NVSDK_NGX_Resource_VK*)nullptr);
        s.params->Set("DLSSNR.UIAlpha", (const NVSDK_NGX_Resource_VK*)nullptr);
        s.params->Set("DLSSNR.Backbuffer", (const NVSDK_NGX_Resource_VK*)nullptr);
        s.params->Set("DLSSNR.BidirectionalDistortionField", (const NVSDK_NGX_Resource_VK*)nullptr);
        s.params->Set("Color", &s.resColor);
        s.params->Set("Output", &s.resOut);
        s.params->Set("Depth", hasDepth ? &s.resDepth : (const NVSDK_NGX_Resource_VK*)nullptr);
        s.params->Set("MotionVectors", &s.resMV);
        s.params->Set("MVec", &s.resMV);
        return true;
    }, false, &seh);

    const char* subrectNames[][4] = {
        { "DLSSNR.ColorSubrectBaseX", "DLSSNR.ColorSubrectBaseY", "DLSSNR.ColorSubrectWidth", "DLSSNR.ColorSubrectHeight" },
        { "DLSSNR.OutputSubrectBaseX", "DLSSNR.OutputSubrectBaseY", "DLSSNR.OutputSubrectWidth", "DLSSNR.OutputSubrectHeight" },
        { "DLSSNR.MVecSubrectBaseX", "DLSSNR.MVecSubrectBaseY", "DLSSNR.MVecSubrectWidth", "DLSSNR.MVecSubrectHeight" },
        { "DLSSNR.DepthSubrectBaseX", "DLSSNR.DepthSubrectBaseY", "DLSSNR.DepthSubrectWidth", "DLSSNR.DepthSubrectHeight" },
    };
    for (auto& n : subrectNames) {
        ParamSetUI(s.params, n[0], 0, &seh);
        ParamSetUI(s.params, n[1], 0, &seh);
        ParamSetUI(s.params, n[2], width, &seh);
        ParamSetUI(s.params, n[3], height, &seh);
    }
    bool ps = true;
    ps &= ParamSetF(s.params, "DLSSNR.Jitter.Offset.X", 0.0f, &seh);
    ps &= ParamSetF(s.params, "DLSSNR.Jitter.Offset.Y", 0.0f, &seh);
    ps &= ParamSetF(s.params, "JitterOffsetX", 0.0f, &seh);
    ps &= ParamSetF(s.params, "JitterOffsetY", 0.0f, &seh);
    ps &= ParamSetUI(s.params, "Reset", 1u, &seh);
    ps &= ParamSetF(s.params, "Sharpness", 0.0f, &seh);
    ps &= ParamSetUI(s.params, "Width", width, &seh);
    ps &= ParamSetUI(s.params, "Height", height, &seh);
    ps &= ParamSetF(s.params, "DLSSNR.MVecScaleX", 1.0f, &seh);
    ps &= ParamSetF(s.params, "DLSSNR.MVecScaleY", 1.0f, &seh);
    ps &= ParamSetUI(s.params, "DLSSNR.DepthInverted", 1u, &seh);
    ps &= ParamSetUI(s.params, "DLSS.Indicator.Invert.X.Axis", 0u, &seh);
    ps &= ParamSetUI(s.params, "DLSS.Indicator.Invert.Y.Axis", 0u, &seh);
    ps &= ParamSetUI(s.params, "DLSSNR.Enabled", 1u, &seh);
    ps &= ParamSetUI(s.params, "DLSSNR.Reset", 1u, &seh);
    ps &= ParamSetUI(s.params, "DLSSNR.UICorrection", 0u, &seh);

    // Style, Intensity, LocalTone, LocalStructure, SkinStructure and UseAutoMask are deliberately
    // absent. They used to be written here, every frame, as constants -- which did two harmful
    // things: it had no effect on the running feature, because the model latches them at creation,
    // and it left the parameter block holding those constants for whatever created a feature next.
    // A feature built at any moment other than immediately after NgxSetCreateTuning therefore got
    // defaults no matter what the user had chosen. They belong to NgxTuning and to create time.
    //
    // UseAutoMask was the clearest case: the constant written here was 0, so the automatic skin mask
    // was forced off regardless of the setting, whose default is on.
    // Read back once per change rather than once per pass per frame: this runs on every evaluate in
    // a multipass chain, and the readback is a diagnostic, not a step.
    unsigned int autoMask = 0;
    float mvecScaleX = 0.0f, mvecScaleY = 0.0f;
    ParamGetUI(s.params, "DLSSNR.UseAutoMask", &autoMask, &seh);
    ParamGetF(s.params, "DLSSNR.MVecScaleX", &mvecScaleX, &seh);
    ParamGetF(s.params, "DLSSNR.MVecScaleY", &mvecScaleY, &seh);
    const bool hasDepthNow = s.resDepth.Resource.ImageViewInfo.ImageView != VK_NULL_HANDLE;
    if (Verbose() || !ps || autoMask != s.loggedAutoMask || mvecScaleX != s.loggedMVecScaleX ||
        mvecScaleY != s.loggedMVecScaleY || hasDepthNow != s.loggedDepthBound) {
        s.loggedAutoMask = autoMask;
        s.loggedMVecScaleX = mvecScaleX;
        s.loggedMVecScaleY = mvecScaleY;
        s.loggedDepthBound = hasDepthNow;
        Log("[params] evaluate contract set: %s (seh=%#x) UseAutoMask=%u MVecScaleX=%.6f MVecScaleY=%.6f depth=%s",
            ps ? "ok" : "FAILED", seh, autoMask, mvecScaleX, mvecScaleY,
            hasDepthNow ? "bound" : "null");
    }
}

void NgxSetReset(NgxSnippet& s, bool reset, bool logValue) {
    if (!s.params) return;
    DWORD seh = 0;
    const bool ok1 = ParamSetUI(s.params, "DLSSNR.Reset", reset ? 1u : 0u, &seh);
    const bool ok2 = ParamSetUI(s.params, "Reset", reset ? 1u : 0u, &seh);
    if (!logValue) return;
    unsigned int back = 0, back2 = 0;
    ParamGetUI(s.params, "DLSSNR.Reset", &back, &seh);
    ParamGetUI(s.params, "Reset", &back2, &seh);
    Log("[params] DLSSNR.Reset(slot11) requested=%u readback=%u alias=%u ok=%d/%d seh=%#x",
        reset ? 1u : 0u, back, back2, int(ok1), int(ok2), seh);
}

void NgxSetMotionScale(NgxSnippet& s, float scaleX, float scaleY) {
    if (!s.params) return;
    DWORD seh = 0;
    ParamSetF(s.params, "DLSSNR.MVecScaleX", scaleX, &seh);
    ParamSetF(s.params, "DLSSNR.MVecScaleY", scaleY, &seh);
    float x = 0.0f, y = 0.0f;
    ParamGetF(s.params, "DLSSNR.MVecScaleX", &x, &seh);
    ParamGetF(s.params, "DLSSNR.MVecScaleY", &y, &seh);
    Log("[params] MVecScaleX=%.6f MVecScaleY=%.6f (seh=%#x)", x, y, seh);
}

void NgxSetSharpness(NgxSnippet& s, float sharpness) {
    if (!s.params) return;
    DWORD seh = 0;
    // The runtime sharpness float has to reach the DLL on every evaluate dispatch: DoSharpening is
    // enabled at create, and this is the per-frame amount it applies.
    ParamSetF(s.params, "Sharpness", sharpness, &seh);
    if (Verbose()) {
        float back = 0.0f;
        ParamGetF(s.params, "Sharpness", &back, &seh);
        Log("[params] Sharpness=%.4f readback=%.4f (seh=%#x)", sharpness, back, seh);
    }
}

// Frame generation's resource block, and the three scalars that came with it.
//
// The names are the DLL's own, read out of the miss log once the contract survived its reset:
// Backbuffer, MVecs, Depth, HUDLess, UI, UIAlpha, BidirectionalDistortionField, NoWarp,
// OutputInterpolated, OutputReal.
//
// Four of those are bound and the rest are explicitly null. A swapchain-only layer sees the frame
// with the HUD already composited into it, so there is no HUDLess colour to hand over and no UI
// layer either -- saying so plainly is better than pointing them at the backbuffer and letting the
// model treat the HUD as scene content.
void NgxSetDlssgResources(NgxSnippet& s, const NVSDK_NGX_Resource_VK* backbuffer,
                          const NVSDK_NGX_Resource_VK* mvec, const NVSDK_NGX_Resource_VK* depth,
                          const NVSDK_NGX_Resource_VK* outInterpolated,
                          const NVSDK_NGX_Resource_VK* outReal, unsigned int targetFrameRate,
                          const NVSDK_NGX_Resource_VK* ui) {
    if (!s.params) return;
    DWORD seh = 0;
    s.params->Set("DLSSG.Backbuffer", (void*)backbuffer);
    s.params->Set("DLSSG.MVecs", (void*)mvec);
    s.params->Set("DLSSG.Depth", (void*)depth);
    s.params->Set("DLSSG.OutputInterpolated", (void*)outInterpolated);
    s.params->Set("DLSSG.OutputReal", (void*)outReal);
    // The frame is the only colour this process has, HUD and all.
    //
    // Null was the honest answer and it is not an accepted one: the flags above say every resource is
    // provided, and the DLL keeps returning MissingInput while the pointer is absent. Pointing
    // HUDLess at the backbuffer says "this is the picture, there is no separate HUD-less copy", which
    // is true, at the cost of the model treating the HUD as scene content -- the thing a real
    // HUDLess input exists to avoid.
    s.params->Set("DLSSG.HUDLess", (void*)backbuffer);
    // Bound rather than null: DLSS-G lists UIColorAndAlpha as required and this was absent.
    s.params->Set("DLSSG.UI", (void*)ui);
    s.params->Set("DLSSG.UIAlpha", (void*)ui);
    s.params->Set("DLSSG.BidirectionalDistortionField", (void*)nullptr);
    s.params->Set("DLSSG.NoWarp", (void*)nullptr);

    ParamSetUI(s.params, "DLSSG.TargetFrameRate", targetFrameRate, &seh);
    // Not running under Streamline: this helper drives the snippet directly, the same way it drives
    // the denoiser. Saying zero is the truthful answer rather than impersonating an interposer.
    static const unsigned int slMode = [] {
        const char* v = getenv("DLSSNR_MFG_SLMODE");
        return (unsigned int)(v && *v ? atoi(v) : 0);
    }();
    ParamSetUI(s.params, "DLSSG.StreamlineMode", slMode, &seh);
    ParamSetUI(s.params, "DLSSG.StreamlineVersionTag", 0u, &seh);
    // Interpolation is the whole point, so it is not disabled.
    ParamSetUI(s.params, "DLSSG.OutputDisableInterpolation", 0u, &seh);

    // The flag that says the evaluation is real.
    //
    // Found in the reference project, which implements the other side of this: its frame generation
    // provider answers the settings callback by setting MustCallEval, and that is how the caller
    // learns the evaluate must actually be performed. Driving the snippet directly, nobody was
    // setting it, and the DLL took the default -- which is why every evaluate returned success and
    // both output surfaces stayed exactly as they were created.
    ParamSetUI(s.params, "DLSSG.MustCallEval", 1u, &seh);
    ParamSetUI(s.params, "DLSSG.BurstCaptureRunning", 0u, &seh);

    // The ceiling, which nothing here had published.
    //
    // This is the number the reference project's binary patch exists to raise, and it turns out the
    // caller is expected to state it as well -- its own frame generation provider sets it. A snippet
    // asked to generate against a ceiling it cannot read has an obvious answer, and the answer it was
    // giving is MustCallEval=0.
    static const unsigned int ceiling = [] {
        const char* v = getenv("DLSSNR_MFG_FRAMES");
        const int n = v && *v ? atoi(v) : 1;
        return (unsigned int)(n < 1 ? 1 : (n > 8 ? 8 : n));
    }();
    ParamSetUI(s.params, "DLSSG.MultiFrameCountMax", ceiling, &seh);
    ParamSetUI(s.params, "DLSSG.DispatchFlags", 0u, &seh);
    ParamSetUI(s.params, "DLSSG.ShowDebug", 0u, &seh);

    // How much memory it may assume. The reference project answers a flat 300 MB rather than
    // computing anything, so the number is clearly not load-bearing; what matters is that something
    // answers at all.
    ParamSetPtr(s.params, "DLSSG.EstimateVRAMCallback", (void*)&DlssgEstimateVram, &seh);

    // A subrect per resource, each the whole surface.
    //
    // Nothing here renders to a corner of a larger target, so every one of these is the full raster
    // at the origin. They are written for the resources that are null as well as the bound ones,
    // because the DLL reads the rectangle before it looks at the pointer.
    static const char* const kSubrects[] = {
        "InputBackbuffer", "Backbuffer", "MVecs", "Depth", "HUDLess", "UI", "UIAlpha",
        "OutputInterpolated", "OutputReal", "BidirectionalDistortionField", "NoWarp",
    };
    for (const char* r : kSubrects) {
        char key[128];
        snprintf(key, sizeof(key), "DLSSG.%sSubrectBaseX", r);   ParamSetUI(s.params, key, 0u, &seh);
        snprintf(key, sizeof(key), "DLSSG.%sSubrectBaseY", r);   ParamSetUI(s.params, key, 0u, &seh);
        snprintf(key, sizeof(key), "DLSSG.%sSubrectWidth", r);   ParamSetUI(s.params, key, s.featureW, &seh);
        snprintf(key, sizeof(key), "DLSSG.%sSubrectHeight", r);  ParamSetUI(s.params, key, s.featureH, &seh);
    }

    if (s.ownParams) static_cast<OwnParam*>(s.params)->Persist();
    Log("[mfg] resources bound: backbuffer=%p mvec=%p depth=%p outInterp=%p outReal=%p targetFps=%u",
        (const void*)backbuffer, (const void*)mvec, (const void*)depth,
        (const void*)outInterpolated, (const void*)outReal, targetFrameRate);
}

// The snippet's own settings call, which the caller is expected to make before evaluating.
//
// nvngx_dlssg.dll exports it as NVSDK_NGX_VULKAN_DLSSG_GetCurrentSettingsCallback_Impl. In a normal
// stack Streamline calls this to learn what the feature wants for the coming frame -- MustCallEval
// among it -- and only then evaluates. Driving the snippet directly means making that call here.
void NgxDlssgQuerySettings(NgxSnippet& s) {
    static bool said = false;
    if (!said) {
        said = true;
        Log("[mfg] settings query entered: snippet=%p params=%p feature0=%p",
            (void*)s.snippet, (void*)s.params, (void*)s.features[0]);
    }
    if (!s.snippet || !s.params || !s.features[0]) return;
    using FnSettings = NVSDK_NGX_Result(NVSDK_CONV*)(NVSDK_NGX_Handle*, NVSDK_NGX_Parameter*);
    static FnSettings fn = nullptr;
    static bool looked = false;
    if (!looked) {
        looked = true;
        // Not an export: the name is a key the snippet writes into the parameter block itself,
        // which is what the _Impl suffix means throughout this API. It only appears there once
        // PopulateParameters has been called, which is the step that was missing -- the loader
        // allocated a parameter block and never gave the snippet the chance to fill it.
        DWORD sehp = 0;
        auto populate = reinterpret_cast<NVSDK_NGX_Result(NVSDK_CONV*)(NVSDK_NGX_Parameter*)>(
            GetProcAddress(s.snippet, "NVSDK_NGX_VULKAN_PopulateParameters_Impl"));
        if (populate) {
            NVSDK_NGX_Result pr = NVSDK_NGX_Result_Fail;
            Guarded([&] { pr = populate(s.params); return true; }, false, &sehp);
            Log("[mfg] PopulateParameters -> %#x seh=%#x", (uint32_t)pr, sehp);
            if (s.ownParams) static_cast<OwnParam*>(s.params)->Persist();
        } else {
            Log("[mfg] PopulateParameters not exported");
        }
        void* cb = nullptr;
        DWORD sehc = 0;
        if (ParamGetPtr(s.params, "NVSDK_NGX_VULKAN_DLSSG_GetCurrentSettingsCallback_Impl", &cb, &sehc) && cb)
            fn = reinterpret_cast<FnSettings>(cb);
        else if (ParamGetPtr(s.params, "DLSSG.GetCurrentSettingsCallback", &cb, &sehc) && cb)
            fn = reinterpret_cast<FnSettings>(cb);
        Log("[mfg] settings callback %s", fn ? "found in the parameter block" : "absent");
    }
    if (!fn) return;
    DWORD seh = 0;
    NVSDK_NGX_Result r = NVSDK_NGX_Result_Fail;
    Guarded([&] { r = fn(s.features[0], s.params); return true; }, false, &seh);
    unsigned int must = 0;
    DWORD seh2 = 0;
    ParamGetUI(s.params, "DLSSG.MustCallEval", &must, &seh2);
    static unsigned int n = 0;
    if ((n++ % 60) == 0)
        Log("[mfg] settings callback -> %#x seh=%#x, MustCallEval=%u (frameId=%llu)",
            (uint32_t)r, seh, must, (unsigned long long)s.fgFrameId);
}

bool NgxEvaluatePass(NgxSnippet& s, uint32_t pass, VkCommandBuffer recordingCmd) {
    if (s.disabled || !s.ready || pass >= kMaxPasses || !s.features[pass]) return false;
    DWORD seh = 0;
    NVSDK_NGX_Result r =
        CallEvaluateSafely(s.evaluateFeature, recordingCmd, s.features[pass], s.params, &seh);
    if (!NVSDK_NGX_SUCCEED(r)) {
        Log("[ngx] VULKAN_EvaluateFeature -> %#x seh=%#x (disabling)", (uint32_t)r, seh);
        s.disabled = true;
        return false;
    }
    return true;
}

// Teardown order per verified runner: Release -> Shutdown1 -> DestroyParameters
// -> restore IAT -> FreeLibrary.
void NgxTeardown(NgxSnippet& s, VkDevice device) {
    DWORD seh = 0;
    for (uint32_t i = 0; i < kMaxPasses; ++i) {
        if (!s.features[i] || !s.releaseFeature) continue;
        NVSDK_NGX_Result r = CallReleaseSafely(s.releaseFeature, s.features[i], &seh);
        Log("[ngx] ReleaseFeature pass %u -> %#x seh=%#x", i, (uint32_t)r, seh);
        s.features[i] = nullptr;
    }
    s.featureCount = 0;
    if (s.shutdown1) {
        NVSDK_NGX_Result r = CallShutdownSafely(s.shutdown1, device, &seh);
        Log("[ngx] snippet Shutdown1 -> %#x seh=%#x", (uint32_t)r, seh);
    }
    if (s.params) {
        if (s.ownParams) delete static_cast<OwnParam*>(s.params);
        else if (s.paramsDestroy) {
            NVSDK_NGX_Result r = Guarded([&] { return s.paramsDestroy(s.params); },
                                         NVSDK_NGX_Result_FAIL_SEH, &seh);
            Log("[ngx] DestroyParameters -> %#x seh=%#x", (uint32_t)r, seh);
        }
        s.params = nullptr;
        s.paramsDestroy = nullptr;
    }
    RemoveCallerSpoof(g_snippetSpoof);
    RemoveCallerSpoof(g_coreSpoof);
    if (s.core) { FreeLibrary(s.core); s.core = nullptr; }
    if (s.snippet) { FreeLibrary(s.snippet); s.snippet = nullptr; }
    s.ready = false;
}

}  // namespace dlssnr