#pragma once
#include "ngx_abi.h"
#include <windows.h>
#include <string>
#include <vulkan/vulkan.h>

namespace dlssnr {

// Owns the nvngx_dlssnr.dll snippet lifecycle: caller-identity IAT spoof,
// core (nvngx.dll) init, parameter block, Feature 18 create/evaluate/teardown.
// All NGX calls are SEH-guarded; any failure latches `disabled` (pass-through).
struct NgxSnippet {
    HMODULE snippet = nullptr;
    HMODULE core = nullptr;
    HMODULE nvapi = nullptr;

    FnVkInitExt initExt = nullptr;
    FnVkInitExt initExt2 = nullptr;
    FnVkInitExt initPlain = nullptr;
    FnVkCreateFeature createFeature = nullptr;
    FnVkEvaluateFeature evaluateFeature = nullptr;
    FnVkReleaseFeature releaseFeature = nullptr;
    FnVkShutdown1 shutdown1 = nullptr;

    NVSDK_NGX_Parameter* params = nullptr;
    FnVkDestroyParameters paramsDestroy = nullptr;
    bool ownParams = false;
    NVSDK_NGX_Handle* feature = nullptr;
    uint32_t featureW = 0, featureH = 0;

    bool ready = false;     // snippet init + CreateFeature(18) succeeded
    bool disabled = false;  // latched failure -> pass-through forever

    // Captured from NVSDK_NGX_VULKAN_GetFeatureRequirements (bit 0 = HDR path).
    unsigned int featureFlags = 0;
    bool hdrCapable = false;
    int loggedPreset = -1;  // NgxSetPreset logs only on change

    // Caller-identity spoof state
    void** iatSlot = nullptr;
    decltype(&GetModuleFileNameW) originalGetModuleFileNameW = nullptr;

    // Resource structs passed to the DLL (must outlive eval calls)
    NVSDK_NGX_Resource_VK resColor{}, resOut{}, resMV{}, resDepth{};

    std::wstring binDir;
};

// Layer module handle (set in DllMain), used as the spoofed caller identity.
extern HMODULE g_layerModule;

bool NgxLoadAndInit(NgxSnippet& s, VkInstance instance, VkPhysicalDevice pd, VkDevice device,
                    uint32_t width, uint32_t height, VkCommandBuffer recordingCmd);
bool NgxCreateFeature(NgxSnippet& s, uint32_t width, uint32_t height, VkCommandBuffer recordingCmd);
void NgxSetResources(NgxSnippet& s, const NVSDK_NGX_Resource_VK& color,
                     const NVSDK_NGX_Resource_VK& out, const NVSDK_NGX_Resource_VK& mv,
                     const NVSDK_NGX_Resource_VK& depth, uint32_t width, uint32_t height);
void NgxSetReset(NgxSnippet& s, bool reset, bool logValue = false);
void NgxSetPreset(NgxSnippet& s, uint32_t preset);
void NgxSetMotionScale(NgxSnippet& s, float scaleX, float scaleY);
void NgxSetStrengths(NgxSnippet& s, float intensity, float localTone,
                     float localStructure, float skinStructure, float sharpness);
bool NgxEvaluate(NgxSnippet& s, VkCommandBuffer recordingCmd);
void NgxTeardown(NgxSnippet& s, VkDevice device);

}  // namespace dlssnr