// dlssnr_helper.exe — Windows-side DLSSNR (Feature 18) service.
// Owns its own Vulkan device + the nvngx_dlssnr.dll snippet (verified
// standalone_runner sequence), waits on the shared-memory frame queue written
// by VK_LAYER_NV_dlssnr, runs the neural pass, and returns processed frames.
#include "ngx_snippet.h"
#include "guard.h"
#include "logging.h"
#include "../common/shm_protocol.h"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <string>
#include <algorithm>
#include <vector>

using namespace dlssnr;

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

static double NowMs() {
    static const LARGE_INTEGER freq = [] {
        LARGE_INTEGER f{};
        QueryPerformanceFrequency(&f);
        return f;
    }();
    LARGE_INTEGER t{};
    QueryPerformanceCounter(&t);
    return double(t.QuadPart) * 1000.0 / double(freq.QuadPart);
}

static inline void CpuYield() {
#if defined(__x86_64__) || defined(__i386__)
    __asm__ __volatile__("pause" ::: "memory");
#else
    Sleep(0);
#endif
}

// ---------------------------------------------------------------------------
// Shared memory transport (must match layer_linux/src/layer.cpp)
// ---------------------------------------------------------------------------
struct ShmMap {
    HANDLE mapping = nullptr;
    HANDLE file = nullptr;
    void* base = nullptr;
    ShmHeader* hdr = nullptr;
    uint8_t* inPixels = nullptr;
    uint8_t* outPixels = nullptr;
};

static bool ShmOpen(ShmMap& s) {
    // DLSSNR_SHM holds the POSIX path (used by the Linux layer); translate to
    // the Wine-visible drive path (Z:\...) for CreateFileW.
    const char* posix = getenv("DLSSNR_SHM");
    std::string p = (posix && *posix) ? posix : ShmDefaultPath();
    std::wstring winPath = L"Z:";
    for (char c : p) winPath += (c == '/') ? L'\\' : (wchar_t)c;
    std::wstring dir = winPath;
    size_t slash = dir.find_last_of(L"\\/");
    if (slash != std::wstring::npos) {
        dir.resize(slash);
        if (!dir.empty()) CreateDirectoryW(dir.c_str(), nullptr);
    }
    s.file = CreateFileW(winPath.c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
                         nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (s.file == INVALID_HANDLE_VALUE) { Log("[helper] open %ls failed (%lu)", winPath.c_str(), GetLastError()); return false; }
    LARGE_INTEGER size;
    size.QuadPart = (LONGLONG)ShmTotalBytes();
    SetFilePointerEx(s.file, size, nullptr, FILE_BEGIN);
    SetEndOfFile(s.file);
    s.mapping = CreateFileMappingW(s.file, nullptr, PAGE_READWRITE, 0, 0, nullptr);
    if (!s.mapping) { Log("[helper] CreateFileMapping failed"); return false; }
    s.base = MapViewOfFile(s.mapping, FILE_MAP_READ | FILE_MAP_WRITE, 0, 0, 0);
    if (!s.base) { Log("[helper] MapViewOfFile failed"); return false; }
    s.hdr = (ShmHeader*)s.base;
    s.inPixels = (uint8_t*)s.base + kHeaderBytes;
    s.outPixels = s.inPixels + kMaxFrame;
    if (s.hdr->magic.load() != kShmMagic || s.hdr->version.load() != kShmVersion ||
        s.hdr->passes.load() == 0) {
        ShmInitDefaults(s.hdr);
    }
    if (s.hdr->quit.load()) Log("[helper] clearing stale quit flag");
    s.hdr->quit.store(0);
    s.hdr->seq_resp.store(s.hdr->seq_req.load());
    s.hdr->controlSeq.fetch_add(1);
    s.hdr->heartbeat.fetch_add(1);
    Log("[helper] shm attached: %ls", winPath.c_str());
    return true;
}

// ---------------------------------------------------------------------------
// Vulkan context (from standalone_runner/main.cpp, verified)
// ---------------------------------------------------------------------------
static HMODULE g_vkModule = nullptr;
static PFN_vkGetInstanceProcAddr g_gipa = nullptr;

#define VK_FN(name) static PFN_##name name = nullptr;
VK_FN(vkCreateInstance) VK_FN(vkDestroyInstance) VK_FN(vkEnumeratePhysicalDevices)
VK_FN(vkGetPhysicalDeviceProperties) VK_FN(vkEnumerateDeviceExtensionProperties)
VK_FN(vkGetPhysicalDeviceQueueFamilyProperties) VK_FN(vkCreateDevice) VK_FN(vkDestroyDevice)
VK_FN(vkGetDeviceQueue) VK_FN(vkCreateCommandPool) VK_FN(vkDestroyCommandPool)
VK_FN(vkAllocateCommandBuffers) VK_FN(vkBeginCommandBuffer) VK_FN(vkEndCommandBuffer)
VK_FN(vkQueueSubmit) VK_FN(vkCreateFence) VK_FN(vkDestroyFence) VK_FN(vkWaitForFences)
VK_FN(vkResetFences) VK_FN(vkCreateImage) VK_FN(vkDestroyImage) VK_FN(vkCreateImageView) VK_FN(vkDestroyImageView)
VK_FN(vkGetImageMemoryRequirements) VK_FN(vkAllocateMemory) VK_FN(vkFreeMemory)
VK_FN(vkMapMemory) VK_FN(vkUnmapMemory) VK_FN(vkBindImageMemory)
VK_FN(vkCreateBuffer) VK_FN(vkDestroyBuffer) VK_FN(vkGetBufferMemoryRequirements)
VK_FN(vkBindBufferMemory) VK_FN(vkCmdCopyBufferToImage) VK_FN(vkCmdCopyImageToBuffer)
VK_FN(vkCmdCopyImage) VK_FN(vkCmdPipelineBarrier) VK_FN(vkDeviceWaitIdle)
#undef VK_FN

struct VkCtx {
    VkInstance instance = nullptr;
    VkPhysicalDevice physical = nullptr;
    VkDevice device = nullptr;
    VkQueue queue = nullptr;
    uint32_t queueFamily = 0;
    VkCommandPool cmdPool = VK_NULL_HANDLE;
    VkCommandBuffer cmdScratch = VK_NULL_HANDLE;
    VkCommandBuffer cmdCreate = VK_NULL_HANDLE;
    VkCommandBuffer cmdEval = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkBuffer uploadStaging = VK_NULL_HANDLE;
    VkBuffer readStaging = VK_NULL_HANDLE;
    VkDeviceMemory uploadMem = VK_NULL_HANDLE;
    VkDeviceMemory readMem = VK_NULL_HANDLE;
    void* uploadMap = nullptr;
    void* readMap = nullptr;
    size_t stagingSize = 0;
};

struct GpuImage {
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    uint32_t width = 0, height = 0;
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageAspectFlags aspect() const {
        return format == VK_FORMAT_R32_SFLOAT ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
    }
};

static uint32_t FindMemoryType(VkCtx& c, uint32_t bits, VkMemoryPropertyFlags want);
static uint32_t FindHostMemoryType(VkCtx& c, uint32_t bits, bool preferCached);
static bool CreateStaging(VkCtx& c, size_t bytes);

static bool HasDeviceExt(VkPhysicalDevice phys, const char* name) {
    uint32_t count = 0;
    vkEnumerateDeviceExtensionProperties(phys, nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> props(count);
    vkEnumerateDeviceExtensionProperties(phys, nullptr, &count, props.data());
    for (auto& p : props) if (!std::strcmp(p.extensionName, name)) return true;
    return false;
}

static bool CreateContext(VkCtx& c) {
    g_vkModule = LoadLibraryA("vulkan-1.dll");
    if (!g_vkModule) { Log("[helper] no vulkan-1.dll"); return false; }
    g_gipa = (PFN_vkGetInstanceProcAddr)GetProcAddress(g_vkModule, "vkGetInstanceProcAddr");
    vkCreateInstance = (PFN_vkCreateInstance)g_gipa(nullptr, "vkCreateInstance");

    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "dlssnr_helper";
    app.apiVersion = VK_API_VERSION_1_3;
    const char* instExts[] = { "VK_KHR_get_physical_device_properties2", "VK_EXT_debug_utils" };
    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;
    ici.enabledExtensionCount = 2;
    ici.ppEnabledExtensionNames = instExts;
    if (vkCreateInstance(&ici, nullptr, &c.instance) != VK_SUCCESS) {
        // debug_utils unavailable: retry without it
        ici.enabledExtensionCount = 1;
        if (vkCreateInstance(&ici, nullptr, &c.instance) != VK_SUCCESS) { Log("[helper] vkCreateInstance failed"); return false; }
    }

#define LOAD(name) name = (PFN_##name)g_gipa(c.instance, #name);
    LOAD(vkDestroyInstance) LOAD(vkEnumeratePhysicalDevices) LOAD(vkGetPhysicalDeviceProperties)
    LOAD(vkEnumerateDeviceExtensionProperties) LOAD(vkGetPhysicalDeviceQueueFamilyProperties)
    LOAD(vkCreateDevice) LOAD(vkDestroyDevice) LOAD(vkGetDeviceQueue) LOAD(vkCreateCommandPool)
    LOAD(vkDestroyCommandPool) LOAD(vkAllocateCommandBuffers) LOAD(vkBeginCommandBuffer)
    LOAD(vkEndCommandBuffer) LOAD(vkQueueSubmit) LOAD(vkCreateFence) LOAD(vkDestroyFence)
    LOAD(vkWaitForFences) LOAD(vkResetFences) LOAD(vkCreateImage) LOAD(vkDestroyImage)
    LOAD(vkCreateImageView) LOAD(vkDestroyImageView) LOAD(vkGetImageMemoryRequirements)
    LOAD(vkAllocateMemory) LOAD(vkFreeMemory) LOAD(vkMapMemory) LOAD(vkUnmapMemory)
    LOAD(vkBindImageMemory) LOAD(vkCreateBuffer) LOAD(vkDestroyBuffer)
    LOAD(vkGetBufferMemoryRequirements) LOAD(vkBindBufferMemory) LOAD(vkCmdCopyBufferToImage)
    LOAD(vkCmdCopyImageToBuffer) LOAD(vkCmdCopyImage) LOAD(vkCmdPipelineBarrier) LOAD(vkDeviceWaitIdle)
#undef LOAD

    uint32_t devCount = 0;
    vkEnumeratePhysicalDevices(c.instance, &devCount, nullptr);
    std::vector<VkPhysicalDevice> phys(devCount);
    vkEnumeratePhysicalDevices(c.instance, &devCount, phys.data());
    for (auto p : phys) {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(p, &props);
        if (props.vendorID != 0x10DE) continue;
        if (!HasDeviceExt(p, "VK_NVX_binary_import") || !HasDeviceExt(p, "VK_NVX_image_view_handle")) continue;
        c.physical = p;
        Log("[helper] device: %s", props.deviceName);
        break;
    }
    if (!c.physical) { Log("[helper] no NVIDIA device with NVX exts"); return false; }

    uint32_t famCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(c.physical, &famCount, nullptr);
    std::vector<VkQueueFamilyProperties> fams(famCount);
    vkGetPhysicalDeviceQueueFamilyProperties(c.physical, &famCount, fams.data());
    for (uint32_t i = 0; i < famCount; ++i)
        if ((fams[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && (fams[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) {
            c.queueFamily = i; break;
        }

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = c.queueFamily;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;
    std::vector<const char*> enabled;
    for (const char* e : { "VK_NVX_binary_import", "VK_NVX_image_view_handle",
                           "VK_KHR_maintenance1", "VK_KHR_maintenance2", "VK_KHR_maintenance3",
                           "VK_KHR_maintenance4", "VK_KHR_buffer_device_address", "VK_KHR_push_descriptor" })
        if (HasDeviceExt(c.physical, e)) enabled.push_back(e);
    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = (uint32_t)enabled.size();
    dci.ppEnabledExtensionNames = enabled.data();
    if (vkCreateDevice(c.physical, &dci, nullptr, &c.device) != VK_SUCCESS) { Log("[helper] vkCreateDevice failed"); return false; }
    vkGetDeviceQueue(c.device, c.queueFamily, 0, &c.queue);

    VkCommandPoolCreateInfo cpci{};
    cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpci.queueFamilyIndex = c.queueFamily;
    if (vkCreateCommandPool(c.device, &cpci, nullptr, &c.cmdPool) != VK_SUCCESS) return false;
    VkCommandBufferAllocateInfo cbai{};
    cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool = c.cmdPool;
    cbai.commandBufferCount = 3;
    VkCommandBuffer bufs[3];
    if (vkAllocateCommandBuffers(c.device, &cbai, bufs) != VK_SUCCESS) return false;
    c.cmdScratch = bufs[0]; c.cmdCreate = bufs[1]; c.cmdEval = bufs[2];
    VkFenceCreateInfo fci{};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if (vkCreateFence(c.device, &fci, nullptr, &c.fence) != VK_SUCCESS) return false;

    // Staging is allocated on the first frame, at that frame's size, rather than at the largest
    // frame the protocol can carry. Every path that needs it grows it on demand already. Reserving
    // the maximum up front cost two host-visible buffers of kMaxFrame each -- which, once the
    // protocol grew to cover a supersampled 4K model raster, is a quarter of a gigabyte of pinned
    // memory for a game that may present at 1080p.
    return true;
}

static uint32_t FindMemoryType(VkCtx& c, uint32_t bits, VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties mp{};
    PFN_vkGetPhysicalDeviceMemoryProperties getMP =
        (PFN_vkGetPhysicalDeviceMemoryProperties)g_gipa(c.instance, "vkGetPhysicalDeviceMemoryProperties");
    getMP(c.physical, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
    return UINT32_MAX;
}

static uint32_t FindHostMemoryType(VkCtx& c, uint32_t bits, bool preferCached) {
    VkPhysicalDeviceMemoryProperties mp{};
    PFN_vkGetPhysicalDeviceMemoryProperties getMP =
        (PFN_vkGetPhysicalDeviceMemoryProperties)g_gipa(c.instance, "vkGetPhysicalDeviceMemoryProperties");
    getMP(c.physical, &mp);
    const VkMemoryPropertyFlags required =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    int best = -1, bestScore = -1;
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        if (!(bits & (1u << i))) continue;
        VkMemoryPropertyFlags f = mp.memoryTypes[i].propertyFlags;
        if ((f & required) != required) continue;
        int score = 0;
        if (f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) score += preferCached ? 100 : 20;
        if (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) score += 10;
        if (f & VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT) score -= 50;
        if (score > bestScore) { bestScore = score; best = (int)i; }
    }
    return best >= 0 ? (uint32_t)best : UINT32_MAX;
}

static bool CreateStaging(VkCtx& c, size_t bytes) {
    if (bytes <= c.stagingSize && c.uploadMap && c.readMap) return true;
    if (c.device) vkDeviceWaitIdle(c.device);
    auto destroy = [&]() {
        if (c.uploadStaging) vkDestroyBuffer(c.device, c.uploadStaging, nullptr);
        if (c.readStaging) vkDestroyBuffer(c.device, c.readStaging, nullptr);
        if (c.uploadMem) vkFreeMemory(c.device, c.uploadMem, nullptr);
        if (c.readMem) vkFreeMemory(c.device, c.readMem, nullptr);
        c.uploadStaging = c.readStaging = nullptr;
        c.uploadMem = c.readMem = nullptr;
        c.uploadMap = c.readMap = nullptr;
        c.stagingSize = 0;
    };
    destroy();

    auto make = [&](VkBuffer& buf, VkDeviceMemory& mem, void** map) {
        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = bytes;
        bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        if (vkCreateBuffer(c.device, &bci, nullptr, &buf) != VK_SUCCESS) return false;
        VkMemoryRequirements req{};
        vkGetBufferMemoryRequirements(c.device, buf, &req);
        VkMemoryAllocateInfo mai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = FindHostMemoryType(c, req.memoryTypeBits, true);
        if (mai.memoryTypeIndex == UINT32_MAX) return false;
        if (vkAllocateMemory(c.device, &mai, nullptr, &mem) != VK_SUCCESS ||
            vkBindBufferMemory(c.device, buf, mem, 0) != VK_SUCCESS) return false;
        return vkMapMemory(c.device, mem, 0, VK_WHOLE_SIZE, 0, map) == VK_SUCCESS;
    };

    if (!make(c.uploadStaging, c.uploadMem, &c.uploadMap) ||
        !make(c.readStaging, c.readMem, &c.readMap)) {
        destroy();
        return false;
    }
    c.stagingSize = bytes;
    return true;
}

static bool CreateImage2D(VkCtx& c, VkFormat fmt, uint32_t w, uint32_t h, GpuImage& out) {
    VkImageCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = fmt;
    ci.extent = { w, h, 1 };
    ci.mipLevels = 1; ci.arrayLayers = 1; ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
               VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (vkCreateImage(c.device, &ci, nullptr, &out.image) != VK_SUCCESS) return false;
    out.format = fmt; out.width = w; out.height = h; out.layout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkMemoryRequirements req{};
    vkGetImageMemoryRequirements(c.device, out.image, &req);
    VkMemoryAllocateInfo mai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = FindMemoryType(c, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (mai.memoryTypeIndex == UINT32_MAX) return false;
    if (vkAllocateMemory(c.device, &mai, nullptr, &out.memory) != VK_SUCCESS) return false;
    if (vkBindImageMemory(c.device, out.image, out.memory, 0) != VK_SUCCESS) return false;
    VkImageViewCreateInfo vi{};
    vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vi.image = out.image;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = fmt;
    vi.subresourceRange = { out.aspect(), 0, 1, 0, 1 };
    return vkCreateImageView(c.device, &vi, nullptr, &out.view) == VK_SUCCESS;
}

static void DestroyImage2D(VkCtx& c, GpuImage& img) {
    if (img.view) vkDestroyImageView(c.device, img.view, nullptr);
    if (img.image) vkDestroyImage(c.device, img.image, nullptr);
    if (img.memory) vkFreeMemory(c.device, img.memory, nullptr);
    img = {};
}

static void TransitionImage(VkCtx& c, VkCommandBuffer cb, GpuImage& img, VkImageLayout dst,
                            VkAccessFlags srcA, VkAccessFlags dstA,
                            VkPipelineStageFlags ss, VkPipelineStageFlags ds) {
    VkImageMemoryBarrier b{};
    b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b.oldLayout = img.layout; b.newLayout = dst;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = img.image;
    b.subresourceRange = { img.aspect(), 0, 1, 0, 1 };
    b.srcAccessMask = srcA; b.dstAccessMask = dstA;
    vkCmdPipelineBarrier(cb, ss, ds, 0, 0, nullptr, 0, nullptr, 1, &b);
    img.layout = dst;
}

static bool BeginCmd(VkCommandBuffer cb) {
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    return vkBeginCommandBuffer(cb, &bi) == VK_SUCCESS;
}

static bool SubmitAndWait(VkCtx& c, VkCommandBuffer cb) {
    if (vkEndCommandBuffer(cb) != VK_SUCCESS) return false;
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1; si.pCommandBuffers = &cb;
    if (vkQueueSubmit(c.queue, 1, &si, c.fence) != VK_SUCCESS) return false;
    if (vkWaitForFences(c.device, 1, &c.fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) return false;
    vkResetFences(c.device, 1, &c.fence);
    return true;
}

static bool UploadMappedPixels(VkCtx& c, GpuImage& img, size_t bytes) {
    if (!c.uploadMap || bytes > c.stagingSize) return false;
    if (!BeginCmd(c.cmdScratch)) return false;
    TransitionImage(c, c.cmdScratch, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
                    VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT);
    VkBufferImageCopy region{};
    region.imageSubresource = { img.aspect(), 0, 0, 1 };
    region.imageExtent = { img.width, img.height, 1 };
    vkCmdCopyBufferToImage(c.cmdScratch, c.uploadStaging, img.image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    return SubmitAndWait(c, c.cmdScratch);
}

static bool UploadPixels(VkCtx& c, GpuImage& img, const void* pixels, size_t bytes) {
    if (bytes > c.stagingSize && !CreateStaging(c, bytes)) return false;
    if (!c.uploadMap) return false;
    if (pixels != c.uploadMap) std::memcpy(c.uploadMap, pixels, bytes);
    return UploadMappedPixels(c, img, bytes);
}

static bool ReadbackPixels(VkCtx& c, GpuImage& img, size_t bytes) {
    if (bytes > c.stagingSize && !CreateStaging(c, bytes)) return false;
    if (!c.readMap) return false;
    if (!BeginCmd(c.cmdEval)) return false;
    TransitionImage(c, c.cmdEval, img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    VkBufferImageCopy region{};
    region.imageSubresource = { img.aspect(), 0, 0, 1 };
    region.imageExtent = { img.width, img.height, 1 };
    vkCmdCopyImageToBuffer(c.cmdEval, img.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           c.readStaging, 1, &region);
    return SubmitAndWait(c, c.cmdEval);
}



// ---------------------------------------------------------------------------
// Per-size neural pipeline state
// ---------------------------------------------------------------------------
static float ClampF(float v, float lo, float hi) {
    if (!(v >= lo)) return lo;
    if (v > hi) return hi;
    return v;
}

struct NeuralState {
    VkCtx vk{};
    NgxSnippet ngx{};

    // The proxy the layer sent, and two surfaces the chain alternates between. Two, not one, because
    // a pass must read the previous pass's answer while writing its own: with a single surface the
    // model would be reading and writing the same image.
    GpuImage colorIn{}, workA{}, workB{}, mv{}, depth{};

    uint32_t w = 0, h = 0;
    bool ready = false;

    // Per-pass state. A pass owns a feature, the tuning that feature was built with, and whether it
    // still owes the model a history reset.
    NgxTuning tuning[kMaxPasses] = {};
    bool passNeedsReset[kMaxPasses] = {};
    uint32_t livePasses = 0;

    // Rebuilds are spaced rather than done at once: back-to-back NGX creation exhausts the driver's
    // latches and the model stops answering until the process restarts.
    uint32_t lastTuningSeq = 0;
    uint64_t frames = 0;
    uint64_t buildAfter = 0;

    uint64_t evaluates = 0;
};

// How long to wait after a change before rebuilding, and between one rebuild and the next.
static constexpr uint64_t kSettleFrames = 30;

static NgxTuning TuningFor(const ShmHeader* h, uint32_t pass) {
    const PassTuning p = ShmResolvePass(h, pass);
    NgxTuning t;
    t.intensity = ClampF(p.intensity, 0.0f, 4.0f);
    t.localTone = ClampF(p.localTone, 0.0f, 4.0f);
    t.localStructure = ClampF(p.localStructure, 0.0f, 4.0f);
    t.skinStructure = ClampF(p.skinStructure, -1.0f, 4.0f);
    t.style = p.style;
    t.preset = p.preset;
    t.autoMask = p.autoMask ? 1u : 0u;
    return t;
}

static void PublishStatus(ShmMap& shm, NeuralState& ns, uint32_t state) {
    if (!shm.hdr) return;
    shm.hdr->helperState.store(state);
    shm.hdr->modelUp.store(ns.ngx.ready && !ns.ngx.disabled ? 1u : 0u);
    shm.hdr->helperFeatures.store(ns.livePasses);
    ShmStore64(shm.hdr->helperFramesLo, shm.hdr->helperFramesHi, ns.evaluates);
}

static bool EnsureNeural(NeuralState& ns, ShmMap& shm, uint32_t w, uint32_t h) {
    if (ns.ready && ns.w == w && ns.h == h) return true;
    if (ns.ngx.disabled) return false;

    if (ns.ngx.snippet) NgxReleaseAllPasses(ns.ngx, ns.vk.device);
    DestroyImage2D(ns.vk, ns.colorIn);
    DestroyImage2D(ns.vk, ns.workA);
    DestroyImage2D(ns.vk, ns.workB);
    DestroyImage2D(ns.vk, ns.mv);
    DestroyImage2D(ns.vk, ns.depth);
    ns.livePasses = 0;

    if (!CreateImage2D(ns.vk, VK_FORMAT_R8G8B8A8_UNORM, w, h, ns.colorIn) ||
        !CreateImage2D(ns.vk, VK_FORMAT_R8G8B8A8_UNORM, w, h, ns.workA) ||
        !CreateImage2D(ns.vk, VK_FORMAT_R8G8B8A8_UNORM, w, h, ns.workB) ||
        !CreateImage2D(ns.vk, VK_FORMAT_R16G16_SFLOAT, w, h, ns.mv) ||
        !CreateImage2D(ns.vk, VK_FORMAT_R32_SFLOAT, w, h, ns.depth)) {
        Log("[helper] image creation failed at %ux%u", w, h);
        ShmStoreString(shm.hdr->helperReasonSeq, shm.hdr->helperReason, kReasonBytes,
                       "could not allocate the model's surfaces");
        return false;
    }

    // The model is given motion vectors and depth because its parameter block requires them. A
    // present-time layer has neither, so they are zero: the model then has no reprojection to do and
    // judges each frame on its own. This is the honest limit of injecting here rather than inside the
    // renderer, and it is why the pass is weaker in motion than OptiScaler's.
    std::vector<uint8_t> zeros(size_t(w) * h * 4, 0);
    if (!UploadPixels(ns.vk, ns.mv, zeros.data(), zeros.size()) ||
        !UploadPixels(ns.vk, ns.depth, zeros.data(), zeros.size())) return false;

    if (!BeginCmd(ns.vk.cmdScratch)) return false;
    TransitionImage(ns.vk, ns.vk.cmdScratch, ns.mv, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    TransitionImage(ns.vk, ns.vk.cmdScratch, ns.depth, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    TransitionImage(ns.vk, ns.vk.cmdScratch, ns.workA, VK_IMAGE_LAYOUT_GENERAL, 0,
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    TransitionImage(ns.vk, ns.vk.cmdScratch, ns.workB, VK_IMAGE_LAYOUT_GENERAL, 0,
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    if (!SubmitAndWait(ns.vk, ns.vk.cmdScratch)) return false;

    // Pass 0's feature is built with pass 0's tuning, here, because the model reads it at create.
    // Pass 0 is created inside NgxLoadAndInit, so its tuning has to be in the parameter block before
    // that call rather than recorded after it. Setting it afterwards is what made pass 0 always come
    // up with defaults while this side believed it had the user's values.
    NgxTuning first = TuningFor(shm.hdr, 0);

    if (!BeginCmd(ns.vk.cmdCreate)) return false;
    bool ok = NgxLoadAndInit(ns.ngx, ns.vk.instance, ns.vk.physical, ns.vk.device, w, h, ns.vk.cmdCreate, first);
    if (!SubmitAndWait(ns.vk, ns.vk.cmdCreate) || !ok) {
        Log("[helper] snippet init/create failed at %ux%u", w, h);
        ShmStoreString(shm.hdr->helperReasonSeq, shm.hdr->helperReason, kReasonBytes,
                       "the model would not initialise; see the helper log");
        ns.ngx.disabled = true;
        PublishStatus(shm, ns, kHelperModelFailed);
        return false;
    }

    ns.tuning[0] = first;
    ns.passNeedsReset[0] = true;
    ns.livePasses = 1;
    ns.w = w;
    ns.h = h;
    ns.ready = true;
    ns.buildAfter = ns.frames + kSettleFrames;
    Log("[helper] neural ready %ux%u", w, h);
    ShmStoreString(shm.hdr->helperReasonSeq, shm.hdr->helperReason, kReasonBytes, "");
    PublishStatus(shm, ns, kHelperRunning);
    return true;
}

// Bind one pass's input and output. The proxy the layer sent is read by pass 0 and never written by
// the chain, so what the composition later differences against is the whole chain's edit rather than
// the last pass's edit against the one before it.
static void BindPass(NeuralState& ns, uint32_t pass, GpuImage*& in, GpuImage*& out) {
    if (pass == 0) {
        in = &ns.colorIn;
        out = &ns.workA;
    } else if (pass % 2 == 1) {
        in = &ns.workA;
        out = &ns.workB;
    } else {
        in = &ns.workB;
        out = &ns.workA;
    }
}

static void FillResource(NVSDK_NGX_Resource_VK& r, GpuImage& img, bool rw) {
    r = {};
    r.Type = NVSDK_NGX_RESOURCE_VK_TYPE_VK_IMAGE_VIEW;
    r.ReadWrite = rw;
    r.Resource.ImageViewInfo.ImageView = img.view;
    r.Resource.ImageViewInfo.Image = img.image;
    r.Resource.ImageViewInfo.SubresourceRange = { img.aspect(), 0, 1, 0, 1 };
    r.Resource.ImageViewInfo.Format = img.format;
    r.Resource.ImageViewInfo.Width = img.width;
    r.Resource.ImageViewInfo.Height = img.height;
}

// Bring the built features into line with what the header asks for.
//
// One build per call and never before the settle window has passed. Both halves matter: NGX creation
// is expensive and back-to-back creation exhausts the driver's latches, after which the model stops
// answering until the process restarts.
static void MaintainPasses(NeuralState& ns, ShmMap& shm, uint32_t wanted) {
    if (!ns.ready || ns.ngx.disabled) return;

    // Has anything the model latches at creation changed?
    //
    // Compared by value rather than by watching tuningSeq. The sequence is a hint, not the truth: a
    // header reset returns it to zero while this process still remembers a larger number, and the
    // change that follows then looks like no change at all. Seven atomic loads per live pass per
    // frame is nothing next to a control that silently stops working.
    {
        ns.lastTuningSeq = shm.hdr->tuningSeq.load();
        bool any = false;
        for (uint32_t i = 0; i < ns.livePasses; ++i) {
            if (TuningFor(shm.hdr, i) != ns.tuning[i]) { any = true; break; }
        }
        if (any) {
            Log("[helper] model tuning changed; rebuilding %u pass(es) after a settle", ns.livePasses);
            vkDeviceWaitIdle(ns.vk.device);
            NgxReleaseAllPasses(ns.ngx, ns.vk.device);
            ns.livePasses = 0;

            // ns.ready stays true on purpose. It means "the snippet is loaded and the surfaces
            // exist", not "a feature is built" -- clearing it would send EnsureNeural back through
            // NgxLoadAndInit, which reloads a 165 MB DLL and re-runs the caller spoof to rebuild
            // something that only needed its features made again. The features come back through the
            // build path below, one at a time.
            ns.ngx.ready = true;
            ns.buildAfter = ns.frames + kSettleFrames;
            return;
        }
    }

    if (wanted < ns.livePasses) {
        vkDeviceWaitIdle(ns.vk.device);
        for (uint32_t i = wanted; i < ns.livePasses; ++i) NgxReleasePass(ns.ngx, i, ns.vk.device);
        ns.livePasses = wanted;
        return;
    }

    // Frames where nothing is built fail open: the layer presents the game's own frame, which is the
    // right answer while the model has no feature to answer with.
    if (wanted > ns.livePasses && ns.frames >= ns.buildAfter) {
        const uint32_t pass = ns.livePasses;
        NgxTuning t = TuningFor(shm.hdr, pass);
        NgxSetCreateTuning(ns.ngx, t);
        if (!BeginCmd(ns.vk.cmdCreate)) return;
        const bool built = NgxCreatePass(ns.ngx, pass, ns.w, ns.h, ns.vk.cmdCreate);
        SubmitAndWait(ns.vk, ns.vk.cmdCreate);
        if (built) {
            ns.tuning[pass] = t;
            ns.passNeedsReset[pass] = true;
            ns.livePasses = pass + 1;
        } else {
            // A later pass failing is a ceiling, not a fault: the chain simply runs at what fits.
            Log("[helper] pass %u would not build; holding the chain at %u", pass, ns.livePasses);
            shm.hdr->helperPassCeiling.store(ns.livePasses);
        }
        ns.buildAfter = ns.frames + kSettleFrames;
    }
}

static bool ProcessFrame(NeuralState& ns, ShmMap& shm) {
    uint32_t w = shm.hdr->width.load(), h = shm.hdr->height.load();
    if (!w || !h || w > kMaxW || h > kMaxH) return false;

    const size_t px = size_t(w) * h;
    const size_t bytes = px * 4;
    ++ns.frames;

    if (!ShmNeuralEnabled(shm.hdr)) {
        std::memcpy(shm.outPixels, shm.inPixels, bytes);
        return true;
    }

    const uint32_t wanted = ShmPasses(shm.hdr);
    const bool time = TimeEnabled();
    const double t0 = NowMs();

    if (!EnsureNeural(ns, shm, w, h)) return false;
    MaintainPasses(ns, shm, wanted);
    if (!ns.ready || ns.livePasses == 0) return false;

    // The proxy the layer encoded. It is already R8G8B8A8_UNORM and display-referred, so there is
    // nothing to swizzle and nothing to convert.
    if (bytes > ns.vk.stagingSize && !CreateStaging(ns.vk, bytes)) return false;
    std::memcpy(ns.vk.uploadMap, shm.inPixels, bytes);
    const double tUpload = time ? NowMs() : 0.0;
    if (!UploadMappedPixels(ns.vk, ns.colorIn, bytes)) return false;

    const uint32_t passes = std::min(wanted, ns.livePasses);
    GpuImage* last = nullptr;

    for (uint32_t pass = 0; pass < passes; ++pass) {
        GpuImage *in = nullptr, *out = nullptr;
        BindPass(ns, pass, in, out);

        NVSDK_NGX_Resource_VK rc{}, ro{}, rm{}, rd{};
        FillResource(rc, *in, false);
        FillResource(ro, *out, true);
        FillResource(rm, ns.mv, false);
        FillResource(rd, ns.depth, false);
        NgxSetResources(ns.ngx, rc, ro, rm, rd, w, h);

        // Sharpness is the one strength the model reads at evaluate, so it follows the setting
        // without a rebuild; everything else was latched when this pass's feature was built.
        const PassTuning ps = ShmResolvePass(shm.hdr, pass);
        NgxSetSharpness(ns.ngx, ClampF(ps.sharpness, 0.0f, 1.0f));

        if (!BeginCmd(ns.vk.cmdEval)) return false;
        TransitionImage(ns.vk, ns.vk.cmdEval, *in, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        TransitionImage(ns.vk, ns.vk.cmdEval, *out, VK_IMAGE_LAYOUT_GENERAL,
            VK_ACCESS_MEMORY_READ_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

        NgxSetReset(ns.ngx, ns.passNeedsReset[pass]);
        ns.passNeedsReset[pass] = false;

        if (!NgxEvaluatePass(ns.ngx, pass, ns.vk.cmdEval)) {
            vkEndCommandBuffer(ns.vk.cmdEval);
            return false;
        }
        if (!SubmitAndWait(ns.vk, ns.vk.cmdEval)) return false;
        last = out;
    }
    const double tEval = time ? NowMs() : 0.0;

    if (!last || !ReadbackPixels(ns.vk, *last, bytes)) return false;
    std::memcpy(shm.outPixels, ns.vk.readMap, bytes);
    const double tDone = time ? NowMs() : 0.0;

    ++ns.evaluates;
    if (shm.hdr) {
        ShmStore64(shm.hdr->helperFramesLo, shm.hdr->helperFramesHi, ns.evaluates);
        shm.hdr->helperEvalMsBits.store(FloatToBits(float(tEval - tUpload)));
        shm.hdr->helperFeatures.store(ns.livePasses);
        shm.hdr->modelUp.store(1);
    }

    if (time) {
        static int frameNo = 0;
        if (++frameNo % TimeInterval() == 0) {
            Log("[time] passes=%u/%u upload=%.2f eval=%.2f readback=%.2f total=%.2f ms",
                passes, wanted, tUpload - t0, tEval - tUpload, tDone - tEval, tDone - t0);
        }
    }
    return true;
}

int main() {
    Log("=== dlssnr_helper starting ===");
    char exePath[MAX_PATH];
    if (GetModuleFileNameA(nullptr, exePath, MAX_PATH) > 0) {
        std::string dir(exePath);
        auto slash = dir.find_last_of("\\/");
        if (slash != std::string::npos) {
            dir.resize(slash);
            SetCurrentDirectoryA(dir.c_str());
        }
    }
    InstallGuard();
    g_layerModule = GetModuleHandleW(nullptr);

    ShmMap shm{};
    if (!ShmOpen(shm)) return 2;

    if (const char* v = getenv("DLSSNR_Passes"); v && *v) {
        uint32_t p = uint32_t(atoi(v));
        if (p >= 1 && p <= kMaxPasses) {
            shm.hdr->passes.store(p);
            shm.hdr->controlSeq.fetch_add(1);
        }
    } else if (const char* v = getenv("DLSSNR_PASSES"); v && *v) {
        uint32_t p = uint32_t(atoi(v));
        if (p >= 1 && p <= kMaxPasses) {
            shm.hdr->passes.store(p);
            shm.hdr->controlSeq.fetch_add(1);
        }
    }

    NeuralState ns{};
    if (!CreateContext(ns.vk)) return 3;
    Log("[helper] context ready, waiting for frames");

    uint32_t lastReq = shm.hdr->seq_resp.load();
    while (!shm.hdr->quit.load()) {
        uint32_t req = shm.hdr->seq_req.load();
        if (req == lastReq) {
            for (int i = 0; i < 20000; ++i) {
                if (shm.hdr->quit.load() || shm.hdr->seq_req.load() != lastReq) break;
                CpuYield();
            }
            if (!shm.hdr->quit.load() && shm.hdr->seq_req.load() == lastReq) {
                shm.hdr->heartbeat.fetch_add(1);
                Sleep(1);
            }
            continue;
        }
        bool ok = ProcessFrame(ns, shm);
        if (!ok) Log("[helper] frame %u failed (w=%u h=%u)", req, shm.hdr->width.load(), shm.hdr->height.load());
        shm.hdr->seq_resp.store(req);
        lastReq = req;
        if (ns.ngx.disabled) break;
    }

    if (shm.hdr->quit.load()) Log("[helper] quit requested");
    else if (ns.ngx.disabled) Log("[helper] neural disabled");
    Log("[helper] shutting down");
    if (ns.ngx.snippet) NgxTeardown(ns.ngx, ns.vk.device);
    vkDeviceWaitIdle(ns.vk.device);
    if (ns.vk.uploadStaging) vkDestroyBuffer(ns.vk.device, ns.vk.uploadStaging, nullptr);
    if (ns.vk.readStaging) vkDestroyBuffer(ns.vk.device, ns.vk.readStaging, nullptr);
    if (ns.vk.uploadMem) vkFreeMemory(ns.vk.device, ns.vk.uploadMem, nullptr);
    if (ns.vk.readMem) vkFreeMemory(ns.vk.device, ns.vk.readMem, nullptr);
    if (ns.vk.device) vkDestroyDevice(ns.vk.device, nullptr);
    if (ns.vk.instance) vkDestroyInstance(ns.vk.instance, nullptr);
    return 0;
}