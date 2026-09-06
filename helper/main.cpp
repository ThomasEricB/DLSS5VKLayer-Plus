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
    std::string p = (posix && *posix) ? posix : "/tmp/dlssnr_shm.bin";
    std::wstring winPath = L"Z:";
    for (char c : p) winPath += (c == '/') ? L'\\' : (wchar_t)c;
    s.file = CreateFileW(winPath.c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
                         nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (s.file == INVALID_HANDLE_VALUE) { Log("[helper] open %ls failed (%lu)", winPath.c_str(), GetLastError()); return false; }
    LARGE_INTEGER size;
    size.QuadPart = (LONGLONG)(4096 + kMaxFrame * 2);
    SetFilePointerEx(s.file, size, nullptr, FILE_BEGIN);
    SetEndOfFile(s.file);
    s.mapping = CreateFileMappingW(s.file, nullptr, PAGE_READWRITE, 0, 0, nullptr);
    if (!s.mapping) { Log("[helper] CreateFileMapping failed"); return false; }
    s.base = MapViewOfFile(s.mapping, FILE_MAP_READ | FILE_MAP_WRITE, 0, 0, 0);
    if (!s.base) { Log("[helper] MapViewOfFile failed"); return false; }
    s.hdr = (ShmHeader*)s.base;
    s.inPixels = (uint8_t*)s.base + 4096;
    s.outPixels = s.inPixels + kMaxFrame;
    if (s.hdr->magic.load() != kShmMagic || s.hdr->passes.load() == 0) {
        ShmInitDefaults(s.hdr);
    }
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
    VkCommandPool cmdPool = nullptr;
    VkCommandBuffer cmdScratch = nullptr;
    VkCommandBuffer cmdCreate = nullptr;
    VkCommandBuffer cmdEval = nullptr;
    VkFence fence = nullptr;
    VkBuffer uploadStaging = nullptr;
    VkBuffer readStaging = nullptr;
    VkDeviceMemory uploadMem = nullptr;
    VkDeviceMemory readMem = nullptr;
    void* uploadMap = nullptr;
    void* readMap = nullptr;
    size_t stagingSize = 0;
};

struct GpuImage {
    VkImage image = nullptr;
    VkImageView view = nullptr;
    VkDeviceMemory memory = nullptr;
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
    return CreateStaging(c, kMaxFrame);
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
struct NeuralState {
    VkCtx vk{};
    NgxSnippet ngx{};
    GpuImage colorIn{}, colorOut{}, mv{}, depth{};
    uint32_t w = 0, h = 0;
    bool ready = false;
    bool firstFrame = true;
};

static void Swizzle(uint8_t* dst, const uint8_t* src, size_t px) {
    uint32_t* d = (uint32_t*)dst;
    const uint32_t* s = (const uint32_t*)src;
    size_t i = 0;
    for (; i + 4 <= px; i += 4) {
        uint32_t v0 = s[i + 0], v1 = s[i + 1], v2 = s[i + 2], v3 = s[i + 3];
        d[i + 0] = (v0 & 0xFF00FF00u) | ((v0 & 0x00FF0000u) >> 16) | ((v0 & 0x000000FFu) << 16);
        d[i + 1] = (v1 & 0xFF00FF00u) | ((v1 & 0x00FF0000u) >> 16) | ((v1 & 0x000000FFu) << 16);
        d[i + 2] = (v2 & 0xFF00FF00u) | ((v2 & 0x00FF0000u) >> 16) | ((v2 & 0x000000FFu) << 16);
        d[i + 3] = (v3 & 0xFF00FF00u) | ((v3 & 0x00FF0000u) >> 16) | ((v3 & 0x000000FFu) << 16);
    }
    for (; i < px; ++i) {
        uint32_t v = s[i];
        d[i] = (v & 0xFF00FF00u) | ((v & 0x00FF0000u) >> 16) | ((v & 0x000000FFu) << 16);
    }
}

static float ClampF(float v, float lo, float hi) {
    if (!(v >= lo)) return lo;
    if (v > hi) return hi;
    return v;
}

static bool EnsureNeural(NeuralState& ns, uint32_t w, uint32_t h) {
    if (ns.ready && ns.w == w && ns.h == h) return true;
    if (ns.ngx.disabled) return false;
    if (ns.ngx.feature) { NgxTeardown(ns.ngx, ns.vk.device); }
    DestroyImage2D(ns.vk, ns.colorIn); DestroyImage2D(ns.vk, ns.colorOut);
    DestroyImage2D(ns.vk, ns.mv); DestroyImage2D(ns.vk, ns.depth);

    if (!CreateImage2D(ns.vk, VK_FORMAT_R8G8B8A8_UNORM, w, h, ns.colorIn) ||
        !CreateImage2D(ns.vk, VK_FORMAT_R8G8B8A8_UNORM, w, h, ns.colorOut) ||
        !CreateImage2D(ns.vk, VK_FORMAT_R16G16_SFLOAT, w, h, ns.mv) ||
        !CreateImage2D(ns.vk, VK_FORMAT_R32_SFLOAT, w, h, ns.depth)) {
        Log("[helper] image creation failed"); return false;
    }
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
    TransitionImage(ns.vk, ns.vk.cmdScratch, ns.colorOut, VK_IMAGE_LAYOUT_GENERAL, 0,
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    if (!SubmitAndWait(ns.vk, ns.vk.cmdScratch)) return false;

    if (!BeginCmd(ns.vk.cmdCreate)) return false;
    bool ok = NgxLoadAndInit(ns.ngx, ns.vk.instance, ns.vk.physical, ns.vk.device, w, h, ns.vk.cmdCreate);
    if (!SubmitAndWait(ns.vk, ns.vk.cmdCreate) || !ok) {
        Log("[helper] snippet init/create failed at %ux%u", w, h);
        ns.ngx.disabled = true;
        return false;
    }

    auto fill = [](NVSDK_NGX_Resource_VK& r, GpuImage& img, bool rw) {
        r = {};
        r.Type = NVSDK_NGX_RESOURCE_VK_TYPE_VK_IMAGE_VIEW;
        r.ReadWrite = rw;
        r.Resource.ImageViewInfo.ImageView = img.view;
        r.Resource.ImageViewInfo.Image = img.image;
        r.Resource.ImageViewInfo.SubresourceRange = { img.aspect(), 0, 1, 0, 1 };
        r.Resource.ImageViewInfo.Format = img.format;
        r.Resource.ImageViewInfo.Width = img.width;
        r.Resource.ImageViewInfo.Height = img.height;
    };
    NVSDK_NGX_Resource_VK rc{}, ro{}, rm{}, rd{};
    fill(rc, ns.colorIn, false); fill(ro, ns.colorOut, true); fill(rm, ns.mv, false); fill(rd, ns.depth, false);
    NgxSetResources(ns.ngx, rc, ro, rm, rd, w, h);

    ns.w = w; ns.h = h; ns.ready = true; ns.firstFrame = true;
    Log("[helper] neural ready %ux%u", w, h);
    return true;
}

static bool ProcessFrame(NeuralState& ns, ShmMap& shm) {
    uint32_t w = shm.hdr->width.load(), h = shm.hdr->height.load();
    uint32_t fmt = shm.hdr->format.load();
    if (!w || !h || w > kMaxW || h > kMaxH) return false;

    size_t px = size_t(w) * h;
    size_t bytes = px * 4;
    if (!ShmNeuralEnabled(shm.hdr)) {
        std::memcpy(shm.outPixels, shm.inPixels, bytes);
        return true;
    }

    const uint32_t passes = ShmPasses(shm.hdr);
    const bool time = TimeEnabled();
    const double t0 = NowMs();
    if (!EnsureNeural(ns, w, h)) return false;

    const uint8_t* in = shm.inPixels;
    if (bytes > ns.vk.stagingSize && !CreateStaging(ns.vk, bytes)) return false;
    if (fmt == 0) Swizzle((uint8_t*)ns.vk.uploadMap, in, px);   // BGRA -> RGBA
    else std::memcpy(ns.vk.uploadMap, in, bytes);
    const double tSwizzleIn = time ? NowMs() : 0.0;
    if (!UploadMappedPixels(ns.vk, ns.colorIn, bytes)) return false;
    const double tUpload = time ? NowMs() : 0.0;

    for (uint32_t pass = 0; pass < passes; ++pass) {
        const PassStrength ps = ShmGetPassStrength(shm.hdr, pass);
        NgxSetStrengths(ns.ngx,
            ClampF(ps.intensity, 0.0f, 4.0f),
            ClampF(ps.localTone, 0.0f, 4.0f),
            ClampF(ps.localStructure, 0.0f, 4.0f),
            ClampF(ps.skinStructure, -1.0f, 4.0f),
            ClampF(ps.sharpness, 0.0f, 1.0f));
        if (!BeginCmd(ns.vk.cmdEval)) return false;

        VkAccessFlags inSrc = 0;
        VkPipelineStageFlags inStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        if (ns.colorIn.layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
            inSrc = VK_ACCESS_TRANSFER_WRITE_BIT; inStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        } else if (ns.colorIn.layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            inSrc = VK_ACCESS_SHADER_READ_BIT; inStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        } else if (ns.colorIn.layout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) {
            inSrc = VK_ACCESS_TRANSFER_READ_BIT; inStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        }
        TransitionImage(ns.vk, ns.vk.cmdEval, ns.colorIn, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            inSrc, VK_ACCESS_SHADER_READ_BIT, inStage, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

        VkAccessFlags outSrc = 0;
        VkPipelineStageFlags outStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        if (ns.colorOut.layout == VK_IMAGE_LAYOUT_GENERAL) {
            outSrc = VK_ACCESS_SHADER_WRITE_BIT; outStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        } else if (ns.colorOut.layout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) {
            outSrc = VK_ACCESS_TRANSFER_READ_BIT; outStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        } else if (ns.colorOut.layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
            outSrc = VK_ACCESS_TRANSFER_WRITE_BIT; outStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        }
        TransitionImage(ns.vk, ns.vk.cmdEval, ns.colorOut, VK_IMAGE_LAYOUT_GENERAL,
            outSrc, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
            outStage, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

        NgxSetReset(ns.ngx, ns.firstFrame && pass == 0);
        if (pass == 0) ns.firstFrame = false;
        if (!NgxEvaluate(ns.ngx, ns.vk.cmdEval)) { vkEndCommandBuffer(ns.vk.cmdEval); return false; }
        if (!SubmitAndWait(ns.vk, ns.vk.cmdEval)) return false;

        if (pass + 1 < passes) {
            if (!BeginCmd(ns.vk.cmdScratch)) return false;
            TransitionImage(ns.vk, ns.vk.cmdScratch, ns.colorOut, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
            TransitionImage(ns.vk, ns.vk.cmdScratch, ns.colorIn, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
            VkImageCopy copy{};
            copy.srcSubresource = { ns.colorOut.aspect(), 0, 0, 1 };
            copy.srcOffset = { 0, 0, 0 };
            copy.dstSubresource = { ns.colorIn.aspect(), 0, 0, 1 };
            copy.dstOffset = { 0, 0, 0 };
            copy.extent = { w, h, 1 };
            vkCmdCopyImage(ns.vk.cmdScratch, ns.colorOut.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           ns.colorIn.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
            if (!SubmitAndWait(ns.vk, ns.vk.cmdScratch)) return false;
        }
    }
    const double tEval = time ? NowMs() : 0.0;

    if (!ReadbackPixels(ns.vk, ns.colorOut, bytes)) return false;
    const double tReadback = time ? NowMs() : 0.0;
    if (fmt == 0) Swizzle(shm.outPixels, (const uint8_t*)ns.vk.readMap, px);  // RGBA -> BGRA
    else std::memcpy(shm.outPixels, ns.vk.readMap, bytes);
    const double tSwizzleOut = time ? NowMs() : 0.0;

    if (time) {
        static int frameNo = 0;
        if (++frameNo % TimeInterval() == 0) {
            Log("[time] passes=%u swizzleIn=%.2f upload=%.2f eval=%.2f readback=%.2f swizzleOut=%.2f total=%.2f ms",
                passes, tSwizzleIn - t0, tUpload - tSwizzleIn, tEval - tUpload,
                tReadback - tEval, tSwizzleOut - tReadback, tSwizzleOut - t0);
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
            if (!shm.hdr->quit.load() && shm.hdr->seq_req.load() == lastReq) Sleep(1);
            continue;
        }
        bool ok = ProcessFrame(ns, shm);
        if (!ok) Log("[helper] frame %u failed (w=%u h=%u)", req, shm.hdr->width.load(), shm.hdr->height.load());
        shm.hdr->seq_resp.store(req);
        lastReq = req;
        if (ns.ngx.disabled) break;
    }

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