// A reproducer for the pipelined path's ghosting.
//
// Why it exists: vkcube spins a small cube against a static background, so the frame as a whole never
// moves. Measured on 40 matched capture pairs, the edit it produces sits exactly on the current
// frame's structure -- peak cross-correlation at zero offset in every frame, at every settle rate. It
// cannot show a stale edit, which is why a run of metrics taken on it reported clean while the real
// artifact was plain on a screen in Half-Life.
//
// This translates the whole picture by a fixed number of pixels per frame, which is what a
// first-person camera does. The displacement is therefore known exactly, so a stale edit has a
// ground-truth offset to be found at, and any claim about ghosting made here is falsifiable.
//
//   build:  see build.sh, target build/pan
//   run:    VKLayer_DLSS5=1 ./build/pan --frames 400 --speed 6
//
// --speed is pixels per frame; 6 at 100 fps is a brisk but ordinary turn.

#include <vulkan/vulkan.h>
#include <xcb/xcb.h>
#include <vulkan/vulkan_xcb.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>

#define CHECK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { \
    fprintf(stderr, "%s:%d: %s -> %d\n", __FILE__, __LINE__, #x, int(r_)); exit(1); } } while (0)

struct PushConstants { float offsetX, offsetY; uint32_t width, height; };

static std::vector<char> ReadFile(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<char> v(static_cast<size_t>(n), 0);
    if (fread(v.data(), 1, size_t(n), f) != size_t(n)) { fprintf(stderr, "short read\n"); exit(1); }
    fclose(f);
    return v;
}

static uint32_t FindMemoryType(VkPhysicalDevice pd, uint32_t bits, VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
    fprintf(stderr, "no memory type\n"); exit(1);
}

int main(int argc, char** argv) {
    uint32_t frames = 400, width = 1280, height = 720;
    float speed = 6.0f;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--frames") && i + 1 < argc) frames = uint32_t(atoi(argv[++i]));
        else if (!strcmp(argv[i], "--speed") && i + 1 < argc) speed = float(atof(argv[++i]));
        else if (!strcmp(argv[i], "--width") && i + 1 < argc) width = uint32_t(atoi(argv[++i]));
        else if (!strcmp(argv[i], "--height") && i + 1 < argc) height = uint32_t(atoi(argv[++i]));
    }

    // ---- window -----------------------------------------------------------
    int screenNum = 0;
    xcb_connection_t* conn = xcb_connect(nullptr, &screenNum);
    if (!conn || xcb_connection_has_error(conn)) { fprintf(stderr, "no X display\n"); return 1; }
    const xcb_setup_t* setup = xcb_get_setup(conn);
    xcb_screen_iterator_t it = xcb_setup_roots_iterator(setup);
    for (int i = 0; i < screenNum; i++) xcb_screen_next(&it);
    xcb_screen_t* screen = it.data;

    xcb_window_t window = xcb_generate_id(conn);
    uint32_t values[2] = { screen->black_pixel, XCB_EVENT_MASK_EXPOSURE | XCB_EVENT_MASK_KEY_PRESS };
    xcb_create_window(conn, XCB_COPY_FROM_PARENT, window, screen->root, 0, 0,
                      uint16_t(width), uint16_t(height), 0, XCB_WINDOW_CLASS_INPUT_OUTPUT,
                      screen->root_visual, XCB_CW_BACK_PIXEL | XCB_CW_EVENT_MASK, values);
    const char* title = "dlssnr pan";
    xcb_change_property(conn, XCB_PROP_MODE_REPLACE, window, XCB_ATOM_WM_NAME, XCB_ATOM_STRING, 8,
                        uint32_t(strlen(title)), title);
    xcb_map_window(conn, window);
    xcb_flush(conn);

    // ---- instance ---------------------------------------------------------
    VkApplicationInfo app{ VK_STRUCTURE_TYPE_APPLICATION_INFO };
    app.pApplicationName = "dlssnr-pan";
    app.apiVersion = VK_API_VERSION_1_2;
    const char* instExts[] = { VK_KHR_SURFACE_EXTENSION_NAME, VK_KHR_XCB_SURFACE_EXTENSION_NAME };
    VkInstanceCreateInfo ici{ VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    ici.pApplicationInfo = &app;
    ici.enabledExtensionCount = 2;
    ici.ppEnabledExtensionNames = instExts;
    VkInstance instance{};
    CHECK(vkCreateInstance(&ici, nullptr, &instance));

    VkSurfaceKHR surface{};
    VkXcbSurfaceCreateInfoKHR sci{ VK_STRUCTURE_TYPE_XCB_SURFACE_CREATE_INFO_KHR };
    sci.connection = conn;
    sci.window = window;
    CHECK(vkCreateXcbSurfaceKHR(instance, &sci, nullptr, &surface));

    // ---- device -----------------------------------------------------------
    uint32_t pdCount = 0;
    vkEnumeratePhysicalDevices(instance, &pdCount, nullptr);
    std::vector<VkPhysicalDevice> pds(pdCount);
    vkEnumeratePhysicalDevices(instance, &pdCount, pds.data());
    VkPhysicalDevice pd = VK_NULL_HANDLE;
    uint32_t queueFamily = 0;
    for (auto cand : pds) {
        uint32_t qf = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(cand, &qf, nullptr);
        std::vector<VkQueueFamilyProperties> qp(qf);
        vkGetPhysicalDeviceQueueFamilyProperties(cand, &qf, qp.data());
        for (uint32_t i = 0; i < qf; i++) {
            VkBool32 present = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(cand, i, surface, &present);
            if ((qp[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && present) { pd = cand; queueFamily = i; break; }
        }
        if (pd) break;
    }
    if (!pd) { fprintf(stderr, "no suitable device\n"); return 1; }

    const float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{ VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
    qci.queueFamilyIndex = queueFamily;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;
    const char* devExts[] = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };
    VkDeviceCreateInfo dci{ VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = 1;
    dci.ppEnabledExtensionNames = devExts;
    VkDevice device{};
    CHECK(vkCreateDevice(pd, &dci, nullptr, &device));
    VkQueue queue{};
    vkGetDeviceQueue(device, queueFamily, 0, &queue);

    // ---- swapchain --------------------------------------------------------
    VkSurfaceCapabilitiesKHR caps{};
    CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(pd, surface, &caps));
    uint32_t fmtCount = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(pd, surface, &fmtCount, nullptr);
    std::vector<VkSurfaceFormatKHR> fmts(fmtCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(pd, surface, &fmtCount, fmts.data());
    VkSurfaceFormatKHR chosen = fmts[0];
    for (auto& f : fmts)
        if (f.format == VK_FORMAT_B8G8R8A8_UNORM || f.format == VK_FORMAT_R8G8B8A8_UNORM) { chosen = f; break; }

    if (caps.currentExtent.width != 0xFFFFFFFFu) { width = caps.currentExtent.width; height = caps.currentExtent.height; }

    VkSwapchainCreateInfoKHR swci{ VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR };
    swci.surface = surface;
    swci.minImageCount = caps.minImageCount < 3 ? 3 : caps.minImageCount;
    if (caps.maxImageCount && swci.minImageCount > caps.maxImageCount) swci.minImageCount = caps.maxImageCount;
    swci.imageFormat = chosen.format;
    swci.imageColorSpace = chosen.colorSpace;
    swci.imageExtent = { width, height };
    swci.imageArrayLayers = 1;
    swci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    swci.preTransform = caps.currentTransform;
    swci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    swci.presentMode = VK_PRESENT_MODE_IMMEDIATE_KHR;
    swci.clipped = VK_TRUE;
    VkSwapchainKHR swapchain{};
    CHECK(vkCreateSwapchainKHR(device, &swci, nullptr, &swapchain));

    uint32_t imgCount = 0;
    vkGetSwapchainImagesKHR(device, swapchain, &imgCount, nullptr);
    std::vector<VkImage> swapImages(imgCount);
    vkGetSwapchainImagesKHR(device, swapchain, &imgCount, swapImages.data());

    // ---- the scene image (storage, blitted into the swapchain) -------------
    VkImageCreateInfo ii{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    ii.imageType = VK_IMAGE_TYPE_2D;
    ii.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    ii.extent = { width, height, 1 };
    ii.mipLevels = 1;
    ii.arrayLayers = 1;
    ii.samples = VK_SAMPLE_COUNT_1_BIT;
    ii.tiling = VK_IMAGE_TILING_OPTIMAL;
    ii.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImage sceneImage{};
    CHECK(vkCreateImage(device, &ii, nullptr, &sceneImage));
    VkMemoryRequirements mr{};
    vkGetImageMemoryRequirements(device, sceneImage, &mr);
    VkMemoryAllocateInfo mai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    mai.allocationSize = mr.size;
    mai.memoryTypeIndex = FindMemoryType(pd, mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VkDeviceMemory sceneMem{};
    CHECK(vkAllocateMemory(device, &mai, nullptr, &sceneMem));
    CHECK(vkBindImageMemory(device, sceneImage, sceneMem, 0));

    VkImageViewCreateInfo vci{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    vci.image = sceneImage;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = ii.format;
    vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    VkImageView sceneView{};
    CHECK(vkCreateImageView(device, &vci, nullptr, &sceneView));

    // ---- pipeline ---------------------------------------------------------
    std::string exe(argv[0]);
    std::string dir = exe.substr(0, exe.find_last_of('/') == std::string::npos ? 0 : exe.find_last_of('/'));
    std::vector<char> spv;
    for (const char* cand : { "pan.comp.spv", "../test_layer/pan.comp.spv", "test_layer/pan.comp.spv" }) {
        std::string p = dir.empty() ? std::string(cand) : dir + "/" + cand;
        FILE* f = fopen(p.c_str(), "rb");
        if (f) { fclose(f); spv = ReadFile(p); break; }
        f = fopen(cand, "rb");
        if (f) { fclose(f); spv = ReadFile(cand); break; }
    }
    if (spv.empty()) { fprintf(stderr, "pan.comp.spv not found\n"); return 1; }

    VkShaderModuleCreateInfo smci{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    smci.codeSize = spv.size();
    smci.pCode = reinterpret_cast<const uint32_t*>(spv.data());
    VkShaderModule module{};
    CHECK(vkCreateShaderModule(device, &smci, nullptr, &module));

    VkDescriptorSetLayoutBinding b{};
    b.binding = 0;
    b.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    b.descriptorCount = 1;
    b.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo dslci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    dslci.bindingCount = 1;
    dslci.pBindings = &b;
    VkDescriptorSetLayout dsl{};
    CHECK(vkCreateDescriptorSetLayout(device, &dslci, nullptr, &dsl));

    VkPushConstantRange pcr{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants) };
    VkPipelineLayoutCreateInfo plci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &dsl;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcr;
    VkPipelineLayout layout{};
    CHECK(vkCreatePipelineLayout(device, &plci, nullptr, &layout));

    VkComputePipelineCreateInfo cpci{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = module;
    cpci.stage.pName = "main";
    cpci.layout = layout;
    VkPipeline pipeline{};
    CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipeline));

    VkDescriptorPoolSize ps{ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1 };
    VkDescriptorPoolCreateInfo dpci{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    dpci.maxSets = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &ps;
    VkDescriptorPool pool{};
    CHECK(vkCreateDescriptorPool(device, &dpci, nullptr, &pool));
    VkDescriptorSetAllocateInfo dsai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    dsai.descriptorPool = pool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &dsl;
    VkDescriptorSet set{};
    CHECK(vkAllocateDescriptorSets(device, &dsai, &set));
    VkDescriptorImageInfo dii{ VK_NULL_HANDLE, sceneView, VK_IMAGE_LAYOUT_GENERAL };
    VkWriteDescriptorSet wds{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
    wds.dstSet = set;
    wds.dstBinding = 0;
    wds.descriptorCount = 1;
    wds.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    wds.pImageInfo = &dii;
    vkUpdateDescriptorSets(device, 1, &wds, 0, nullptr);

    // ---- command buffers and sync -----------------------------------------
    VkCommandPoolCreateInfo cpi{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpi.queueFamilyIndex = queueFamily;
    VkCommandPool cmdPool{};
    CHECK(vkCreateCommandPool(device, &cpi, nullptr, &cmdPool));

    const uint32_t kInFlight = 2;
    std::vector<VkCommandBuffer> cbs(kInFlight);
    VkCommandBufferAllocateInfo cbai{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    cbai.commandPool = cmdPool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = kInFlight;
    CHECK(vkAllocateCommandBuffers(device, &cbai, cbs.data()));

    std::vector<VkSemaphore> acquired(kInFlight), rendered(kInFlight);
    std::vector<VkFence> fences(kInFlight);
    for (uint32_t i = 0; i < kInFlight; i++) {
        VkSemaphoreCreateInfo si{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
        CHECK(vkCreateSemaphore(device, &si, nullptr, &acquired[i]));
        CHECK(vkCreateSemaphore(device, &si, nullptr, &rendered[i]));
        VkFenceCreateInfo fi{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        CHECK(vkCreateFence(device, &fi, nullptr, &fences[i]));
    }

    const auto barrier = [&](VkCommandBuffer cb, VkImage image, VkImageLayout from, VkImageLayout to,
                             VkAccessFlags srcAccess, VkAccessFlags dstAccess) {
        VkImageMemoryBarrier bar{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
        bar.srcAccessMask = srcAccess;
        bar.dstAccessMask = dstAccess;
        bar.oldLayout = from;
        bar.newLayout = to;
        bar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bar.image = image;
        bar.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &bar);
    };

    printf("pan: %ux%u, %.1f px/frame, %u frames\n", width, height, speed, frames);
    fflush(stdout);

    uint32_t slot = 0;
    for (uint32_t frame = 0; frame < frames; frame++) {
        CHECK(vkWaitForFences(device, 1, &fences[slot], VK_TRUE, UINT64_MAX));
        CHECK(vkResetFences(device, 1, &fences[slot]));

        uint32_t index = 0;
        VkResult acq = vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, acquired[slot], VK_NULL_HANDLE, &index);
        if (acq == VK_ERROR_OUT_OF_DATE_KHR) break;

        VkCommandBuffer cb = cbs[slot];
        CHECK(vkResetCommandBuffer(cb, 0));
        VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        CHECK(vkBeginCommandBuffer(cb, &bi));

        barrier(cb, sceneImage, frame < kInFlight ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_IMAGE_LAYOUT_GENERAL, 0, VK_ACCESS_SHADER_WRITE_BIT);

        PushConstants pc{};
        // Diagonal, so a ghost is displaced in both axes and cannot be confused with a scanline effect.
        pc.offsetX = speed * float(frame);
        pc.offsetY = speed * 0.5f * float(frame);
        pc.width = width;
        pc.height = height;
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, nullptr);
        vkCmdPushConstants(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cb, (width + 7) / 8, (height + 7) / 8, 1);

        barrier(cb, sceneImage, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        barrier(cb, swapImages[index], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                0, VK_ACCESS_TRANSFER_WRITE_BIT);

        VkImageBlit blit{};
        blit.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        blit.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        blit.srcOffsets[1] = { int32_t(width), int32_t(height), 1 };
        blit.dstOffsets[1] = { int32_t(width), int32_t(height), 1 };
        vkCmdBlitImage(cb, sceneImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, swapImages[index],
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_NEAREST);

        barrier(cb, swapImages[index], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                VK_ACCESS_TRANSFER_WRITE_BIT, 0);
        CHECK(vkEndCommandBuffer(cb));

        VkPipelineStageFlags wait = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        VkSubmitInfo si{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
        si.waitSemaphoreCount = 1;
        si.pWaitSemaphores = &acquired[slot];
        si.pWaitDstStageMask = &wait;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cb;
        si.signalSemaphoreCount = 1;
        si.pSignalSemaphores = &rendered[slot];
        CHECK(vkQueueSubmit(queue, 1, &si, fences[slot]));

        VkPresentInfoKHR pi{ VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };
        pi.waitSemaphoreCount = 1;
        pi.pWaitSemaphores = &rendered[slot];
        pi.swapchainCount = 1;
        pi.pSwapchains = &swapchain;
        pi.pImageIndices = &index;
        VkResult pr = vkQueuePresentKHR(queue, &pi);
        if (pr == VK_ERROR_OUT_OF_DATE_KHR) break;

        slot = (slot + 1) % kInFlight;

        while (xcb_generic_event_t* ev = xcb_poll_for_event(conn)) free(ev);
    }

    vkDeviceWaitIdle(device);
    printf("pan: done\n");
    return 0;
}
