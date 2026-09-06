// VK_LAYER_NV_dlssnr — Linux-side Vulkan layer (loaded by the host loader,
// including the one inside Wine/winevulkan). Hooks the swapchain lifecycle;
// on present, ships the frame to the Windows DLSSNR helper over shared
// memory and presents the neural-processed result.
//
// Enabled implicitly via enable_environment DLSSNR_ENABLE=1 (winevulkan
// rejects explicitly-named layers, so implicit enable is required under Wine).
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <vulkan/vulkan.h>
#include <vulkan/vk_layer.h>

#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <chrono>
#include <unordered_map>
#include <vector>

#include "../../common/shm_protocol.h"
#include "composition.h"
#include "vk_table.h"

#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define VK_LAYER_NAME "VK_LAYER_NV_dlssnr"

// ---------------------------------------------------------------------------
// Logging
// ---------------------------------------------------------------------------
static void Log(const char* fmt, ...) {
    static FILE* f = [] {
        const char* p = getenv("DLSSNR_LOG");
        return p && *p ? fopen(p, "a") : stderr;
    }();
    char buf[2048];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    // One mutex for the sink, not one per line: the previous form allocated a fresh
    // std::mutex on every call and leaked it, which at present rates is a leak per frame.
    static std::mutex sinkMutex;
    std::lock_guard<std::mutex> lk(sinkMutex);
    fprintf(f, "[dlssnr-layer] %s\n", buf);
    fflush(f);
}

static bool TimeEnabled() {
    static const bool v = [] {
        const char* p = getenv("DLSSNR_TIME");
        return p && p[0] == '1';
    }();
    return v;
}

static int TimeInterval() {
    static const int v = [] {
        const char* p = getenv("DLSSNR_TIME_EVERY");
        return p && *p ? atoi(p) : 30;
    }();
    return v > 0 ? v : 30;
}

static inline double NowMs() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static inline void CpuYield() {
#if defined(__x86_64__) || defined(__i386__)
    __asm__ __volatile__("pause" ::: "memory");
#else
    std::this_thread::yield();
#endif
}

// ---------------------------------------------------------------------------
// Shared memory transport
// ---------------------------------------------------------------------------
struct ShmMap {
    void* base = nullptr;
    ShmHeader* hdr = nullptr;
    uint8_t* inPixels = nullptr;
    uint8_t* outPixels = nullptr;
    uint32_t seq = 0;
    uint32_t timeouts = 0;
    uint32_t lastControlSeq = 0;
    uint32_t lastHeartbeat = 0;
    bool dead = false;
};

static void EnsureParentDir(const std::string& path) {
    size_t slash = path.find_last_of('/');
    if (slash == std::string::npos || slash == 0) return;
    std::string dir = path.substr(0, slash);
    size_t pos = 1;
    while ((pos = dir.find('/', pos)) != std::string::npos) {
        std::string part = dir.substr(0, pos);
        mkdir(part.c_str(), 0700);
        pos += 1;
    }
    mkdir(dir.c_str(), 0700);
}

static std::string ShmDefaultPathLayer() {
    const char* rt = getenv("XDG_RUNTIME_DIR");
    if (rt && *rt) return std::string(rt) + "/dlssnr/shm.bin";
    const char* uid = getenv("DLSSNR_UID");
    if (uid && *uid) return std::string("/tmp/dlssnr-") + uid + "/shm.bin";
    return std::string("/tmp/dlssnr-") + std::to_string(getuid()) + "/shm.bin";
}

static bool ShmOpen(ShmMap& s) {
    if (s.base) return true;
    const char* path = getenv("DLSSNR_SHM");
    std::string p = (path && *path) ? path : ShmDefaultPathLayer();
    EnsureParentDir(p);
    int fd = open(p.c_str(), O_RDWR | O_CREAT, 0600);
    if (fd < 0) { Log("[shm] open %s failed", p.c_str()); return false; }
    size_t total = ShmTotalBytes();
    struct stat st{};
    if (fstat(fd, &st) != 0 || (size_t)st.st_size < total) {
        if (ftruncate(fd, (off_t)total) != 0) { close(fd); return false; }
    }
    void* m = mmap(nullptr, total, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (m == MAP_FAILED) { Log("[shm] mmap failed"); return false; }
    s.base = m;
    s.hdr = (ShmHeader*)m;
    s.inPixels = (uint8_t*)m + kHeaderBytes;
    s.outPixels = s.inPixels + kMaxFrame;
    // A mapping left by an older build has a different magic, a different version, or a header
    // laid out differently; re-initialising is the only safe reading of any of those.
    if (s.hdr->magic.load() != kShmMagic || s.hdr->version.load() != kShmVersion ||
        s.hdr->passes.load() == 0) {
        ShmInitDefaults(s.hdr);
    }
    s.lastHeartbeat = s.hdr->heartbeat.load();
    Log("[shm] attached %s seq_req=%u seq_resp=%u", p.c_str(),
        s.hdr->seq_req.load(), s.hdr->seq_resp.load());
    return true;
}

static bool ShmNeuralEnabled(ShmMap& s) {
    if (!ShmOpen(s)) return true;
    if (s.hdr->quit.load()) { s.dead = true; return false; }
    const uint32_t ctrl = s.hdr->controlSeq.load();
    if (ctrl != s.lastControlSeq) {
        s.lastControlSeq = ctrl;
        if (s.dead && ::ShmNeuralEnabled(s.hdr)) {
            s.dead = false;
            s.timeouts = 0;
            Log("[shm] control changed, re-enabling");
        }
    }
    const uint32_t hb = s.hdr->heartbeat.load();
    if (hb != s.lastHeartbeat) {
        s.lastHeartbeat = hb;
        if (s.dead && ::ShmNeuralEnabled(s.hdr)) {
            s.dead = false;
            s.timeouts = 0;
            Log("[shm] helper heartbeat, re-enabling");
        }
    }
    if (s.dead) return false;
    return ::ShmNeuralEnabled(s.hdr);
}

// One round trip: publish the proxy, wait for the model's answer, copy it back.
//
// What crosses is the proxy at the model's own resolution, always R8G8B8A8_UNORM and always
// display-referred, because the encode has already done that work on the GPU. The helper therefore
// never has to know what format the game presents in, and the working scale reduces this copy
// quadratically -- which on this transport is the difference the setting actually buys.
static bool ShmProcessFrame(ShmMap& s, uint32_t w, uint32_t h, const void* proxy, void* modelOut) {
    if (s.dead) return false;
    if (!ShmOpen(s)) { s.dead = true; return false; }
    if (w > kMaxW || h > kMaxH) return false;
    if (s.hdr->quit.load()) { s.dead = true; return false; }

    const bool time = TimeEnabled();
    const double t0 = NowMs();
    const size_t bytes = size_t(w) * h * 4;
    std::memcpy(s.inPixels, proxy, bytes);
    const double tCopy = NowMs();
    s.hdr->width.store(w);
    s.hdr->height.store(h);
    s.hdr->format.store(1u);  // the encode always writes RGBA order
    uint32_t req = s.hdr->seq_req.load() + 1;
    s.hdr->seq_req.store(req);

    // Wait for the helper (fail-open: present the original frame on timeout).
    const double tSignal = NowMs();
    for (;;) {
        if (s.hdr->seq_resp.load() >= req) {
            s.timeouts = 0;
            std::memcpy(modelOut, s.outPixels, bytes);
            if (time) {
                static int frameNo = 0;
                if (++frameNo % TimeInterval() == 0) {
                    const double tDone = NowMs();
                    Log("[time] shm copy=%.2f signal=%.2f wait=%.2f total=%.2f ms",
                        tCopy - t0, tSignal - tCopy, tDone - tSignal, tDone - t0);
                }
            }
            return true;
        }
        if (s.hdr->quit.load()) { s.dead = true; return false; }
        const double elapsed = NowMs() - tSignal;
        if (elapsed >= 1000.0) break;
        if (elapsed < 2.0) CpuYield();
        else std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
    if (++s.timeouts >= 8) { s.dead = true; Log("[shm] helper unresponsive, disabling"); }
    return false;
}

// ---------------------------------------------------------------------------
// Dispatch chains
// ---------------------------------------------------------------------------
struct InstanceChain {
    PFN_vkGetInstanceProcAddr next_gipa = nullptr;

    // The instance-level entry points the composition needs, resolved once. Kept here rather than on
    // the device chain because this is where the VkInstance handle is in scope.
    dlssnr::InstanceTable table;

    PFN_vkDestroyInstance vkDestroyInstance = nullptr;
    PFN_vkEnumeratePhysicalDevices vkEnumeratePhysicalDevices = nullptr;
    PFN_vkGetPhysicalDeviceProperties vkGetPhysicalDeviceProperties = nullptr;
    PFN_vkEnumerateDeviceExtensionProperties vkEnumerateDeviceExtensionProperties = nullptr;
    PFN_vkGetPhysicalDeviceMemoryProperties vkGetPhysicalDeviceMemoryProperties = nullptr;
};

#define DEVICE_FN_LIST(X) \
    X(vkDestroyDevice) X(vkGetDeviceQueue) X(vkGetDeviceQueue2) X(vkCreateSwapchainKHR) X(vkDestroySwapchainKHR) \
    X(vkGetSwapchainImagesKHR) X(vkQueuePresentKHR) X(vkQueueSubmit) X(vkCreateCommandPool) \
    X(vkDestroyCommandPool) X(vkAllocateCommandBuffers) X(vkBeginCommandBuffer) X(vkEndCommandBuffer) \
    X(vkCreateFence) X(vkDestroyFence) X(vkWaitForFences) X(vkResetFences) \
    X(vkCreateImage) X(vkDestroyImage) X(vkGetImageMemoryRequirements) X(vkAllocateMemory) \
    X(vkFreeMemory) X(vkBindImageMemory) X(vkCreateImageView) X(vkDestroyImageView) \
    X(vkMapMemory) X(vkUnmapMemory) X(vkCreateBuffer) X(vkDestroyBuffer) \
    X(vkGetBufferMemoryRequirements) X(vkBindBufferMemory) X(vkCmdCopyBufferToImage) \
    X(vkCmdCopyImageToBuffer) X(vkCmdPipelineBarrier) X(vkDeviceWaitIdle)

struct SwapchainState {
    std::vector<VkImage> images;
    VkFormat format = VK_FORMAT_UNDEFINED;
    uint32_t width = 0, height = 0;
    VkFence fence = nullptr;
    VkCommandPool pool = nullptr;
    VkCommandBuffer cb = nullptr;
    bool ready = false;
    bool passThrough = false;

    // The pass. Owns every surface it needs, including the two host-visible buffers the round trip
    // reads and writes, which is why there are no staging buffers left here.
    std::unique_ptr<dlssnr::Composition> comp;
};

struct DeviceChain {
    InstanceChain* instance = nullptr;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice self = VK_NULL_HANDLE;
    PFN_vkGetDeviceProcAddr next_dpa = nullptr;

    // The same entry points again, in the form the composition takes them.
    dlssnr::DeviceTable table;

    // The loader's hook for installing a dispatch table on a dispatchable object a layer creates.
    // Handed to every layer in its own VkLayerDeviceCreateInfo node; see Hook_CreateDevice.
    PFN_vkSetDeviceLoaderData setDeviceLoaderData = nullptr;
#define X(name) PFN_##name name = nullptr;
    DEVICE_FN_LIST(X)
#undef X
    std::atomic<bool> inert{false};
    std::mutex lock;
    std::unordered_map<VkSwapchainKHR, SwapchainState> swapchains;
    std::unordered_map<VkQueue, uint32_t> queueFamilies;
    ShmMap shm;
    uint64_t framesComposed = 0;
};

static std::unordered_map<VkInstance, InstanceChain> g_instances;
static std::unordered_map<VkPhysicalDevice, InstanceChain*> g_phys;
static std::unordered_map<VkDevice, DeviceChain*> g_devices;
static std::mutex g_stateMutex;

// Where this copy of the layer was loaded from, for the duplicate check below.
static std::string LayerObjectPath() {
    Dl_info info{};
    if (dladdr((const void*)&LayerObjectPath, &info) && info.dli_fname && *info.dli_fname)
        return info.dli_fname;
    return std::string();
}

// True when a *different* copy of this layer is already in the chain.
//
// build.sh installs an implicit-layer manifest pointing at the build tree while install.sh installs
// another pointing at the install prefix, and the loader honours both: two copies of the layer, two
// present hooks, two full round trips, and a single shared-memory file with two writers racing on
// one sequence number. Only the first copy stays live; the rest declare themselves inert and pass
// everything through, which turns a corrupted picture or a hang into one warning line.
//
// The claim is the object's own path rather than a bare flag, so a second call into the same copy --
// which is legal, the loader may negotiate more than once -- is told apart from a second copy.
static bool DuplicateLayerCopy() {
    static const bool dup = [] {
        const std::string self = LayerObjectPath();
        const char* claimed = getenv("DLSSNR_LAYER_OBJECT");
        if (claimed && *claimed) {
            if (self.empty() || self == claimed) return false;
            Log("[layer] another copy is already loaded from %s; this copy (%s) stays inert. "
                "Remove one of the implicit-layer manifests.", claimed, self.c_str());
            return true;
        }
        if (!self.empty()) setenv("DLSSNR_LAYER_OBJECT", self.c_str(), 0);
        return false;
    }();
    return dup;
}

static bool LayerEnabled() {
    static const bool e = [] {
        if (DuplicateLayerCopy()) return false;
        const char* v = getenv("VKLayer_DLSS5");
        if (v && v[0] == '1') return true;
        const char* o = getenv("DLSSNR_ENABLE");
        return o && o[0] == '1';
    }();
    return e;
}

// ---------------------------------------------------------------------------
// Instance hooks
// ---------------------------------------------------------------------------
static VKAPI_ATTR VkResult VKAPI_CALL Hook_CreateInstance(
    const VkInstanceCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator,
    VkInstance* pInstance) {
    auto* link = const_cast<VkLayerInstanceCreateInfo*>((const VkLayerInstanceCreateInfo*)pCreateInfo->pNext);
    while (link && !(link->sType == VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO &&
                     link->function == VK_LAYER_LINK_INFO))
        link = (VkLayerInstanceCreateInfo*)link->pNext;
    if (!link || !link->u.pLayerInfo) return VK_ERROR_INITIALIZATION_FAILED;

    PFN_vkGetInstanceProcAddr next_gipa = link->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    auto create = (PFN_vkCreateInstance)next_gipa(VK_NULL_HANDLE, "vkCreateInstance");
    if (!create) return VK_ERROR_INITIALIZATION_FAILED;

    // Documented pattern: keep the link node in pNext (layers below need it)
    // and advance u.pLayerInfo so the next layer resolves its own chain entry.
    link->u.pLayerInfo = link->u.pLayerInfo->pNext;
    VkResult res = create(pCreateInfo, pAllocator, pInstance);
    if (res != VK_SUCCESS) return res;

    InstanceChain chain{};
    chain.next_gipa = next_gipa;
    chain.vkDestroyInstance = (PFN_vkDestroyInstance)next_gipa(*pInstance, "vkDestroyInstance");
    chain.vkEnumeratePhysicalDevices = (PFN_vkEnumeratePhysicalDevices)next_gipa(*pInstance, "vkEnumeratePhysicalDevices");
    chain.vkGetPhysicalDeviceProperties = (PFN_vkGetPhysicalDeviceProperties)next_gipa(*pInstance, "vkGetPhysicalDeviceProperties");
    chain.vkEnumerateDeviceExtensionProperties = (PFN_vkEnumerateDeviceExtensionProperties)next_gipa(*pInstance, "vkEnumerateDeviceExtensionProperties");
    chain.vkGetPhysicalDeviceMemoryProperties = (PFN_vkGetPhysicalDeviceMemoryProperties)next_gipa(*pInstance, "vkGetPhysicalDeviceMemoryProperties");

    chain.table.next_gipa = next_gipa;
    chain.table.Load(*pInstance);

    std::lock_guard<std::mutex> lk(g_stateMutex);
    g_instances[*pInstance] = chain;
    Log("[layer] vkCreateInstance -> %p", (void*)*pInstance);
    return VK_SUCCESS;
}

static VKAPI_ATTR void VKAPI_CALL Hook_DestroyInstance(VkInstance instance,
                                                       const VkAllocationCallbacks* pAllocator) {
    std::lock_guard<std::mutex> lk(g_stateMutex);
    auto it = g_instances.find(instance);
    if (it == g_instances.end()) return;
    auto destroy = it->second.vkDestroyInstance;
    InstanceChain* chain = &it->second;
    g_instances.erase(it);
    for (auto pit = g_phys.begin(); pit != g_phys.end();)
        pit = (pit->second == chain) ? g_phys.erase(pit) : std::next(pit);
    if (destroy) destroy(instance, pAllocator);
}

static VKAPI_ATTR VkResult VKAPI_CALL Hook_EnumeratePhysicalDevices(
    VkInstance instance, uint32_t* pCount, VkPhysicalDevice* pPhysicalDevices) {
    InstanceChain* chain = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_stateMutex);
        auto it = g_instances.find(instance);
        if (it != g_instances.end()) chain = &it->second;
    }
    if (!chain || !chain->vkEnumeratePhysicalDevices) return VK_ERROR_INITIALIZATION_FAILED;
    VkResult res = chain->vkEnumeratePhysicalDevices(instance, pCount, pPhysicalDevices);
    if (res == VK_SUCCESS && pPhysicalDevices) {
        std::lock_guard<std::mutex> lk(g_stateMutex);
        for (uint32_t i = 0; i < *pCount; ++i) g_phys[pPhysicalDevices[i]] = chain;
    }
    return res;
}

// ---------------------------------------------------------------------------
// Device hooks
// ---------------------------------------------------------------------------
static VKAPI_ATTR VkResult VKAPI_CALL Hook_CreateDevice(
    VkPhysicalDevice physicalDevice, const VkDeviceCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator, VkDevice* pDevice) {
    auto* link = const_cast<VkLayerDeviceCreateInfo*>((const VkLayerDeviceCreateInfo*)pCreateInfo->pNext);
    while (link && !(link->sType == VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO &&
                     link->function == VK_LAYER_LINK_INFO))
        link = (VkLayerDeviceCreateInfo*)link->pNext;
    if (!link || !link->u.pLayerInfo) return VK_ERROR_INITIALIZATION_FAILED;

    PFN_vkGetInstanceProcAddr next_gipa = link->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    PFN_vkGetDeviceProcAddr next_dpa = link->u.pLayerInfo->pfnNextGetDeviceProcAddr;
    auto create = (PFN_vkCreateDevice)next_gipa(VK_NULL_HANDLE, "vkCreateDevice");
    if (!create) return VK_ERROR_INITIALIZATION_FAILED;

    // A second node in the same chain carries vkSetDeviceLoaderData. Every dispatchable object this
    // layer allocates has to be passed through it; see SetLoaderData below for why.
    PFN_vkSetDeviceLoaderData setLoaderData = nullptr;
    for (const auto* n = (const VkLayerDeviceCreateInfo*)pCreateInfo->pNext; n;
         n = (const VkLayerDeviceCreateInfo*)n->pNext) {
        if (n->sType == VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO &&
            n->function == VK_LOADER_DATA_CALLBACK) {
            setLoaderData = n->u.pfnSetDeviceLoaderData;
            break;
        }
    }

    InstanceChain* ic = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_stateMutex);
        auto it = g_phys.find(physicalDevice);
        if (it != g_phys.end()) ic = it->second;
    }

    link->u.pLayerInfo = link->u.pLayerInfo->pNext;
    VkResult res = create(physicalDevice, pCreateInfo, pAllocator, pDevice);
    if (res != VK_SUCCESS) return res;

    DeviceChain* dc = new DeviceChain();
    dc->instance = ic;
    dc->physical = physicalDevice;
    dc->self = *pDevice;
    dc->next_dpa = next_dpa;
    dc->setDeviceLoaderData = setLoaderData;
#define X(name) dc->name = (PFN_##name)next_dpa(*pDevice, #name);
    DEVICE_FN_LIST(X)
#undef X
    dc->table.next_dpa = next_dpa;
    dc->table.Load(*pDevice);
    if (!dc->vkQueuePresentKHR || !dc->vkCreateSwapchainKHR || !ic) dc->inert = true;

    // Neural Rendering is an NGX feature and the helper only ever creates its own device on an
    // NVIDIA GPU, so on anything else there is nothing for this layer to do but cost a round trip.
    // Hybrid machines are the case that matters: an implicit layer is loaded for every device the
    // loader builds, including the integrated one a game may well be running on.
    char deviceName[VK_MAX_PHYSICAL_DEVICE_NAME_SIZE] = "?";
    if (ic && ic->vkGetPhysicalDeviceProperties) {
        VkPhysicalDeviceProperties props{};
        ic->vkGetPhysicalDeviceProperties(physicalDevice, &props);
        std::snprintf(deviceName, sizeof(deviceName), "%s", props.deviceName);
        if (props.vendorID != 0x10DE) {
            dc->inert = true;
            Log("[layer] inert on non-NVIDIA device (vendor %#x): %s", props.vendorID, deviceName);
        }
    }

    std::lock_guard<std::mutex> lk(g_stateMutex);
    g_devices[*pDevice] = dc;
    Log("[layer] vkCreateDevice -> %p on %s (inert=%d enabled=%d)", (void*)*pDevice, deviceName,
        (int) dc->inert.load(), (int) LayerEnabled());
    return VK_SUCCESS;
}

static VKAPI_ATTR void VKAPI_CALL Hook_DestroyDevice(VkDevice device,
                                                     const VkAllocationCallbacks* pAllocator) {
    DeviceChain* dc = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_stateMutex);
        auto it = g_devices.find(device);
        if (it != g_devices.end()) { dc = it->second; g_devices.erase(it); }
    }
    if (!dc) return;
    if (dc->vkDeviceWaitIdle) dc->vkDeviceWaitIdle(device);
    {
        std::lock_guard<std::mutex> lk(dc->lock);
        for (auto& kv : dc->swapchains) {
            SwapchainState& sc = kv.second;
            sc.comp.reset();
            if (sc.fence) dc->vkDestroyFence(device, sc.fence, nullptr);
            if (sc.pool) dc->vkDestroyCommandPool(device, sc.pool, nullptr);
        }
        dc->swapchains.clear();
    }
    if (dc->vkDestroyDevice) dc->vkDestroyDevice(device, pAllocator);
    delete dc;
}

static DeviceChain* FindDevice(VkDevice device) {
    std::lock_guard<std::mutex> lk(g_stateMutex);
    auto it = g_devices.find(device);
    return it == g_devices.end() ? nullptr : it->second;
}

static void RememberQueue(DeviceChain* dc, VkQueue queue, uint32_t family) {
    if (!queue) return;
    std::lock_guard<std::mutex> lk(dc->lock);
    dc->queueFamilies[queue] = family;
}

static VKAPI_ATTR void VKAPI_CALL Hook_GetDeviceQueue(VkDevice device, uint32_t family,
                                                      uint32_t index, VkQueue* pQueue) {
    DeviceChain* dc = FindDevice(device);
    if (!dc || !dc->vkGetDeviceQueue) return;
    dc->vkGetDeviceQueue(device, family, index, pQueue);
    RememberQueue(dc, *pQueue, family);
}

// The 1.1 way of asking for a queue, and the only way to reach one created with
// VkDeviceQueueCreateFlags. A game that uses it never registered its queue through the hook above,
// so the present path could not tell which family the queue belonged to and fell back to family
// zero -- which is the family the command pool was then created on, and need not be the queue's.
static VKAPI_ATTR void VKAPI_CALL Hook_GetDeviceQueue2(VkDevice device,
                                                       const VkDeviceQueueInfo2* pQueueInfo,
                                                       VkQueue* pQueue) {
    DeviceChain* dc = FindDevice(device);
    if (!dc || !dc->vkGetDeviceQueue2) return;
    dc->vkGetDeviceQueue2(device, pQueueInfo, pQueue);
    if (pQueueInfo) RememberQueue(dc, *pQueue, pQueueInfo->queueFamilyIndex);
}

// ---------------------------------------------------------------------------
// Swapchain
// ---------------------------------------------------------------------------
// Which swapchain formats the pass can work in. Every one of them has a UNORM twin the composition
// uses internally; the ten-bit and float entries are new here, and are what lets an HDR game reach
// the model at all -- the encode is exactly the step that turns open-ended light into the kind of
// picture the model was trained on.
static bool SupportedFormat(VkFormat f) {
    return dlssnr::CompositionFormat(f) != VK_FORMAT_UNDEFINED;
}

static VKAPI_ATTR VkResult VKAPI_CALL Hook_CreateSwapchainKHR(
    VkDevice device, const VkSwapchainCreateInfoKHR* pCreateInfo,
    const VkAllocationCallbacks* pAllocator, VkSwapchainKHR* pSwapchain) {
    DeviceChain* dc = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_stateMutex);
        auto it = g_devices.find(device);
        if (it != g_devices.end()) dc = it->second;
    }
    if (!dc || !dc->vkCreateSwapchainKHR) return VK_ERROR_INITIALIZATION_FAILED;

    VkSwapchainCreateInfoKHR m = *pCreateInfo;
    if (!dc->inert && LayerEnabled() && SupportedFormat(pCreateInfo->imageFormat))
        m.imageUsage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    VkResult res = dc->vkCreateSwapchainKHR(device, &m, pAllocator, pSwapchain);
    if (res != VK_SUCCESS || dc->inert || !LayerEnabled()) return res;

    uint32_t count = 0;
    dc->vkGetSwapchainImagesKHR(device, *pSwapchain, &count, nullptr);
    std::vector<VkImage> images(count);
    dc->vkGetSwapchainImagesKHR(device, *pSwapchain, &count, images.data());

    SwapchainState sc{};
    sc.images = std::move(images);
    sc.format = pCreateInfo->imageFormat;
    sc.width = pCreateInfo->imageExtent.width;
    sc.height = pCreateInfo->imageExtent.height;
    sc.passThrough = !SupportedFormat(sc.format) || sc.width > kMaxW || sc.height > kMaxH;

    std::lock_guard<std::mutex> lk(dc->lock);
    Log("[layer] swapchain %p %ux%u fmt=%d passThrough=%d%s", (void*)*pSwapchain,
        pCreateInfo->imageExtent.width, pCreateInfo->imageExtent.height,
        (int)pCreateInfo->imageFormat, (int)sc.passThrough,
        sc.passThrough ? (SupportedFormat(sc.format) ? " (too large)" : " (unsupported format)") : "");
    dc->swapchains[*pSwapchain] = std::move(sc);
    return VK_SUCCESS;
}

static VKAPI_ATTR void VKAPI_CALL Hook_DestroySwapchainKHR(VkDevice device,
    VkSwapchainKHR swapchain, const VkAllocationCallbacks* pAllocator) {
    DeviceChain* dc = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_stateMutex);
        auto it = g_devices.find(device);
        if (it != g_devices.end()) dc = it->second;
    }
    if (!dc) return;
    std::unique_lock<std::mutex> lk(dc->lock);
    auto it = dc->swapchains.find(swapchain);
    if (it != dc->swapchains.end()) {
        lk.unlock();
        if (dc->vkDeviceWaitIdle) dc->vkDeviceWaitIdle(device);
        lk.lock();
        SwapchainState& sc = it->second;
        sc.comp.reset();
        if (sc.fence) dc->vkDestroyFence(device, sc.fence, nullptr);
        if (sc.pool) dc->vkDestroyCommandPool(device, sc.pool, nullptr);
        dc->swapchains.erase(it);
    }
    lk.unlock();
    if (dc->vkDestroySwapchainKHR) dc->vkDestroySwapchainKHR(device, swapchain, pAllocator);
}

// ---------------------------------------------------------------------------
// Present-time neural round-trip
// ---------------------------------------------------------------------------
// Give a dispatchable object this layer allocated the dispatch table the loader expects on it.
//
// VkCommandBuffer and VkQueue are dispatchable: their first word points at a dispatch table, and
// every layer below reads it to find its own state for that object. The loader fills that word in
// for objects the application allocates through the trampoline -- but a layer that allocates one by
// calling straight down the chain bypasses the trampoline, so the loader never sees it and the word
// keeps whatever the ICD left there. The loader hands each layer vkSetDeviceLoaderData precisely so
// it can do that fill-in itself, and calling it is mandatory, not advisory.
//
// Skipping it is invisible with no other layer present: the next call goes straight to the driver,
// which does not read the word. Add any second layer -- Steam's overlay, MangoHud, validation -- and
// that layer reads the word, finds the ICD's loader magic instead of a table, and aborts. Validation
// says so out loud: 'The VkDevice dispatch handle was not found and Validation will crash.'
static bool SetLoaderData(DeviceChain* dc, void* object) {
    if (!dc->setDeviceLoaderData) return true;  // no loader in the chain; nothing to fill in
    return dc->setDeviceLoaderData(dc->self, object) == VK_SUCCESS;
}

static bool CreateResources(DeviceChain* dc, SwapchainState& sc, uint32_t family) {
    VkDevice d = dc->self;

    VkCommandPoolCreateInfo cpci{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpci.queueFamilyIndex = family;
    if (dc->vkCreateCommandPool(d, &cpci, nullptr, &sc.pool) != VK_SUCCESS) return false;
    VkCommandBufferAllocateInfo cbai{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    cbai.commandPool = sc.pool; cbai.commandBufferCount = 1;
    if (dc->vkAllocateCommandBuffers(d, &cbai, &sc.cb) != VK_SUCCESS) return false;
    if (!SetLoaderData(dc, sc.cb)) {
        Log("[layer] vkSetDeviceLoaderData failed for the present command buffer");
        return false;
    }
    VkFenceCreateInfo fci{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    if (dc->vkCreateFence(d, &fci, nullptr, &sc.fence) != VK_SUCCESS) return false;

    if (!dc->instance) return false;
    sc.comp = std::make_unique<dlssnr::Composition>(&dc->table, &dc->instance->table, d, dc->physical);
    if (!sc.comp->Usable()) {
        Log("[layer] composition unavailable: %s", sc.comp->Reason());
        sc.comp.reset();
        return false;
    }
    return true;
}

// Every Vulkan result on the present path, looked at rather than collapsed into a bool.
//
// A failure here was previously indistinguishable from 'nothing to do': the call returned false, the
// caller presented the original frame, and the next frame tried exactly the same thing again. That is
// the right answer for a transient failure and the wrong one for VK_ERROR_DEVICE_LOST, where the
// device is gone, every subsequent call will fail the same way, and the log fills with it.
//
// Losing the device also latches the layer inert, because after that point the fail-open path is the
// only correct one and it costs nothing to take it directly.
static bool NoteVk(DeviceChain* dc, VkResult r, const char* what) {
    if (r == VK_SUCCESS) return true;
    if (r == VK_ERROR_DEVICE_LOST) {
        if (!dc->inert.exchange(true)) Log("[layer] %s -> DEVICE_LOST; layer inert for this device", what);
        return false;
    }
    static std::atomic<uint32_t> reported{0};
    if (reported.fetch_add(1) < 8) Log("[layer] %s -> %d", what, (int) r);
    return false;
}

// Returns true if the swapchain image now holds the composed frame.
//
// Three steps around one round trip. The pass encodes a proxy of the frame on the GPU, that proxy
// crosses to the helper and comes back as the model's answer, and the pass composes the answer onto
// the frame. Between them the work is fenced on the CPU, which is what makes the split possible at
// all: the model is in another process and there is nothing to wait on but a sequence number.
//
// The caller's present semaphores are consumed by the first submit, because that submit is the first
// thing to touch the image. They are therefore unsignalled by the time this returns and must not be
// handed to vkQueuePresentKHR again; the caller presents with none.
//
// Every path out leaves the swapchain image in PRESENT_SRC_KHR, including the ones that give up.
static bool ProcessPresent(DeviceChain* dc, SwapchainState& sc, VkQueue queue,
                           VkImage swapchainImage, uint32_t waitCount,
                           const VkSemaphore* waitSemaphores) {
    if (!sc.comp) return false;
    VkDevice d = dc->self;
    VkCommandBuffer cb = sc.cb;
    const bool time = TimeEnabled();
    const double t0 = time ? NowMs() : 0.0;

    const dlssnr::FrameSettings fs = dlssnr::FrameSettings::Read(dc->shm.hdr);
    const bool linearHdr =
        dlssnr::ColourIsLinearHdr(sc.format, dc->shm.hdr ? dc->shm.hdr->colourMode.load() : kColourAuto);

    if (!sc.comp->Prepare(sc.width, sc.height, sc.format, fs, linearHdr)) {
        Log("[layer] composition cannot run here: %s", sc.comp->Reason());
        return false;
    }

    // A capture is asked for by writing a frame count into the header; taking it clears the request,
    // so one press produces one run rather than one per frame for as long as nobody clears it.
    if (dc->shm.hdr) {
        if (const uint32_t frames = dc->shm.hdr->captureRequest.exchange(0); frames > 0)
            sc.comp->RequestCapture(std::min<uint32_t>(frames, 64));
    }

    VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VkSubmitInfo si{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;

    // Only the first submit waits: the later one is ordered behind it on the same queue and fenced
    // besides, and a binary semaphore may be waited on once per signal.
    std::vector<VkPipelineStageFlags> waitStages(waitCount, VK_PIPELINE_STAGE_TRANSFER_BIT);
    si.waitSemaphoreCount = waitCount;
    si.pWaitSemaphores = waitCount ? waitSemaphores : nullptr;
    si.pWaitDstStageMask = waitCount ? waitStages.data() : nullptr;
    const auto dropWaits = [&] {
        si.waitSemaphoreCount = 0;
        si.pWaitSemaphores = nullptr;
        si.pWaitDstStageMask = nullptr;
    };

    const auto runLeg = [&]() {
        if (!NoteVk(dc, dc->vkEndCommandBuffer(cb), "vkEndCommandBuffer")) return false;
        if (!NoteVk(dc, dc->vkQueueSubmit(queue, 1, &si, sc.fence), "vkQueueSubmit")) return false;
        if (!NoteVk(dc, dc->vkWaitForFences(d, 1, &sc.fence, VK_TRUE, UINT64_MAX), "vkWaitForFences"))
            return false;
        dc->vkResetFences(d, 1, &sc.fence);
        return true;
    };

    // ---- leg 1: the frame the model is shown ----
    if (!NoteVk(dc, dc->vkBeginCommandBuffer(cb, &bi), "vkBeginCommandBuffer")) return false;
    if (!sc.comp->RecordCapture(cb, swapchainImage, fs)) {
        dc->vkEndCommandBuffer(cb);
        return false;
    }
    if (!runLeg()) return false;
    dropWaits();
    sc.comp->ConsumeMeter();
    const double tCapture = time ? NowMs() : 0.0;

    // ---- the round trip ----
    if (!ShmProcessFrame(dc->shm, sc.comp->ModelWidth(), sc.comp->ModelHeight(), sc.comp->ProxyPixels(),
                         sc.comp->ModelPixels())) {
        // Fail-open. Leg 1 already put the image back in PRESENT_SRC_KHR, so the original frame is
        // what gets presented and nothing else is owed.
        return false;
    }
    sc.comp->MarkModelFrame();
    const double tHelper = time ? NowMs() : 0.0;

    // ---- leg 2: the answer, composed back ----
    if (!NoteVk(dc, dc->vkBeginCommandBuffer(cb, &bi), "vkBeginCommandBuffer")) return false;
    if (!sc.comp->RecordCompose(cb, swapchainImage, fs)) {
        dc->vkEndCommandBuffer(cb);
        return false;
    }
    if (!runLeg()) return false;
    sc.comp->WriteCapturedFrame();
    const double tReturn = time ? NowMs() : 0.0;

    if (dc->shm.hdr) {
        ShmStore64(dc->shm.hdr->layerFramesLo, dc->shm.hdr->layerFramesHi, ++dc->framesComposed);
        dc->shm.hdr->layerWidth.store(sc.width);
        dc->shm.hdr->layerHeight.store(sc.height);
        dc->shm.hdr->layerFormat.store(uint32_t(sc.format));
        dc->shm.hdr->layerCompositionUp.store(1);
        dc->shm.hdr->layerMsBits.store(FloatToBits(float(tReturn - t0)));
        dc->shm.hdr->layerMeasuredWhiteBits.store(FloatToBits(sc.comp->MeasuredWhitePoint()));
        dc->shm.hdr->layerHeartbeat.fetch_add(1);
    }

    if (time) {
        static int frameNo = 0;
        if (++frameNo % TimeInterval() == 0) {
            Log("[time] encode=%.2f helper=%.2f resolve=%.2f total=%.2f ms (model %ux%u)",
                tCapture - t0, tHelper - tCapture, tReturn - tHelper, tReturn - t0,
                sc.comp->ModelWidth(), sc.comp->ModelHeight());
        }
    }
    return true;
}

static VKAPI_ATTR VkResult VKAPI_CALL Hook_QueuePresentKHR(VkQueue queue,
                                                           const VkPresentInfoKHR* pPresentInfo) {
    DeviceChain* dc = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_stateMutex);
        if (g_devices.size() == 1) dc = g_devices.begin()->second;
        else {
            for (auto& kv : g_devices) {
                std::lock_guard<std::mutex> dl(kv.second->lock);
                if (kv.second->queueFamilies.count(queue)) { dc = kv.second; break; }
            }
        }
    }
    if (!dc || !dc->vkQueuePresentKHR) return VK_ERROR_INITIALIZATION_FAILED;

    // Whether this call's wait semaphores have already been consumed by a submit of ours. They are
    // handed to the first swapchain we actually process; every path after that presents with none,
    // because a semaphore signalled once may only be waited on once. Presenting with them a second
    // time is a wait that never completes -- which is what a second layer in the chain, Steam's
    // overlay among them, turns from a latent bug into a hang.
    bool waitsConsumed = false;

    if (!dc->inert && LayerEnabled()) {
        std::lock_guard<std::mutex> lk(dc->lock);
        if (!ShmNeuralEnabled(dc->shm)) return dc->vkQueuePresentKHR(queue, pPresentInfo);
        uint32_t family = 0;
        auto qit = dc->queueFamilies.find(queue);
        if (qit != dc->queueFamilies.end()) family = qit->second;
        for (uint32_t i = 0; i < pPresentInfo->swapchainCount; ++i) {
            auto sit = dc->swapchains.find(pPresentInfo->pSwapchains[i]);
            if (sit == dc->swapchains.end()) continue;
            SwapchainState& sc = sit->second;
            if (sc.passThrough || pPresentInfo->pImageIndices[i] >= sc.images.size()) continue;
            if (!sc.ready && !dc->shm.dead) {
                if (!CreateResources(dc, sc, family)) {
                    Log("[layer] staging resources failed for swapchain %p (%ux%u, family %u); "
                        "passing this swapchain through",
                        (void*)pPresentInfo->pSwapchains[i], sc.width, sc.height, family);
                    sc.passThrough = true;
                    continue;
                }
                sc.ready = true;
            }
            if (!sc.ready || dc->shm.dead) continue;
            const uint32_t waitCount = waitsConsumed ? 0u : pPresentInfo->waitSemaphoreCount;
            waitsConsumed = true;
            ProcessPresent(dc, sc, queue, sc.images[pPresentInfo->pImageIndices[i]], waitCount,
                           pPresentInfo->pWaitSemaphores);
            // On failure we simply present the original frame (fail-open). The semaphores are still
            // consumed -- the capture submit waits on them before anything can fail -- so the flag
            // stays set and the present below still drops them.
        }
    }

    if (!waitsConsumed) return dc->vkQueuePresentKHR(queue, pPresentInfo);

    // pNext is carried through untouched: present ids, present timing and the rest belong to the
    // caller and none of them are about semaphores.
    VkPresentInfoKHR pi = *pPresentInfo;
    pi.waitSemaphoreCount = 0;
    pi.pWaitSemaphores = nullptr;
    return dc->vkQueuePresentKHR(queue, &pi);
}

// ---------------------------------------------------------------------------
// Loader entry points
// ---------------------------------------------------------------------------
static PFN_vkVoidFunction LookupHook(const char* n) {
    if (!std::strcmp(n, "vkCreateInstance")) return (PFN_vkVoidFunction)Hook_CreateInstance;
    if (!std::strcmp(n, "vkDestroyInstance")) return (PFN_vkVoidFunction)Hook_DestroyInstance;
    if (!std::strcmp(n, "vkEnumeratePhysicalDevices")) return (PFN_vkVoidFunction)Hook_EnumeratePhysicalDevices;
    if (!std::strcmp(n, "vkCreateDevice")) return (PFN_vkVoidFunction)Hook_CreateDevice;
    if (!std::strcmp(n, "vkDestroyDevice")) return (PFN_vkVoidFunction)Hook_DestroyDevice;
    if (!std::strcmp(n, "vkGetDeviceQueue")) return (PFN_vkVoidFunction)Hook_GetDeviceQueue;
    if (!std::strcmp(n, "vkGetDeviceQueue2")) return (PFN_vkVoidFunction)Hook_GetDeviceQueue2;
    if (!std::strcmp(n, "vkCreateSwapchainKHR")) return (PFN_vkVoidFunction)Hook_CreateSwapchainKHR;
    if (!std::strcmp(n, "vkDestroySwapchainKHR")) return (PFN_vkVoidFunction)Hook_DestroySwapchainKHR;
    if (!std::strcmp(n, "vkQueuePresentKHR")) return (PFN_vkVoidFunction)Hook_QueuePresentKHR;
    return nullptr;
}

static PFN_vkVoidFunction LookupDeviceHook(const char* n) {
    if (!std::strcmp(n, "vkDestroyDevice")) return (PFN_vkVoidFunction)Hook_DestroyDevice;
    if (!std::strcmp(n, "vkGetDeviceQueue")) return (PFN_vkVoidFunction)Hook_GetDeviceQueue;
    if (!std::strcmp(n, "vkGetDeviceQueue2")) return (PFN_vkVoidFunction)Hook_GetDeviceQueue2;
    if (!std::strcmp(n, "vkCreateSwapchainKHR")) return (PFN_vkVoidFunction)Hook_CreateSwapchainKHR;
    if (!std::strcmp(n, "vkDestroySwapchainKHR")) return (PFN_vkVoidFunction)Hook_DestroySwapchainKHR;
    if (!std::strcmp(n, "vkQueuePresentKHR")) return (PFN_vkVoidFunction)Hook_QueuePresentKHR;
    return nullptr;
}

extern "C" {

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance instance, const char* pName);
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetDeviceProcAddr(VkDevice device, const char* pName);

VKAPI_ATTR VkResult VKAPI_CALL
vkNegotiateLoaderLayerInterfaceVersion(VkNegotiateLayerInterface* v) {
    if (!v || v->sType != LAYER_NEGOTIATE_INTERFACE_STRUCT) return VK_ERROR_INITIALIZATION_FAILED;
    if (v->loaderLayerInterfaceVersion > 7) v->loaderLayerInterfaceVersion = 7;
    if (v->loaderLayerInterfaceVersion >= 2) {
        v->pfnGetInstanceProcAddr = vkGetInstanceProcAddr;
        v->pfnGetDeviceProcAddr = vkGetDeviceProcAddr;
        v->pfnGetPhysicalDeviceProcAddr = nullptr;
    }
    Log("=== VK_LAYER_NV_dlssnr loaded (VKLayer_DLSS5=%s) ===", getenv("VKLayer_DLSS5") ? getenv("VKLayer_DLSS5") : "(unset)");
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateInstanceLayerProperties(uint32_t* pCount,
                                                                  VkLayerProperties* pProperties) {
    if (!pCount) return VK_SUCCESS;
    if (!pProperties) { *pCount = 1; return VK_SUCCESS; }
    if (*pCount < 1) { *pCount = 1; return VK_INCOMPLETE; }
    std::memset(pProperties, 0, sizeof(*pProperties));
    std::strncpy(pProperties->layerName, VK_LAYER_NAME, VK_MAX_EXTENSION_NAME_SIZE - 1);
    std::strncpy(pProperties->description, "DLSS 5 Neural Rendering injection layer",
                 VK_MAX_DESCRIPTION_SIZE - 1);
    pProperties->specVersion = VK_MAKE_VERSION(1, 3, 0);
    pProperties->implementationVersion = 1;
    *pCount = 1;
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateInstanceExtensionProperties(const char*, uint32_t* pCount,
                                                                      VkExtensionProperties*) {
    if (pCount) *pCount = 0;
    return VK_SUCCESS;
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance instance, const char* pName) {
    if (!pName) return nullptr;
    if (!std::strcmp(pName, "vkGetInstanceProcAddr")) return (PFN_vkVoidFunction)vkGetInstanceProcAddr;
    if (!std::strcmp(pName, "vkGetDeviceProcAddr")) return (PFN_vkVoidFunction)vkGetDeviceProcAddr;
    if (!std::strcmp(pName, "vkNegotiateLoaderLayerInterfaceVersion"))
        return (PFN_vkVoidFunction)vkNegotiateLoaderLayerInterfaceVersion;
    if (!std::strcmp(pName, "vkEnumerateInstanceLayerProperties"))
        return (PFN_vkVoidFunction)vkEnumerateInstanceLayerProperties;
    if (!std::strcmp(pName, "vkEnumerateInstanceExtensionProperties"))
        return (PFN_vkVoidFunction)vkEnumerateInstanceExtensionProperties;
    if (auto fn = LookupHook(pName)) return fn;
    if (instance) {
        std::lock_guard<std::mutex> lk(g_stateMutex);
        auto it = g_instances.find(instance);
        if (it != g_instances.end() && it->second.next_gipa) return it->second.next_gipa(instance, pName);
    }
    return nullptr;
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetDeviceProcAddr(VkDevice device, const char* pName) {
    if (!pName) return nullptr;
    if (!std::strcmp(pName, "vkGetDeviceProcAddr")) return (PFN_vkVoidFunction)vkGetDeviceProcAddr;
    if (auto fn = LookupDeviceHook(pName)) return fn;
    if (device) {
        std::lock_guard<std::mutex> lk(g_stateMutex);
        auto it = g_devices.find(device);
        if (it != g_devices.end() && it->second->next_dpa) return it->second->next_dpa(device, pName);
    }
    return nullptr;
}

}  // extern "C"