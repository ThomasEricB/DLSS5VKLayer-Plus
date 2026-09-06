#include "composition.h"
#include "log.h"

#include <algorithm>
#include <cmath>

namespace dlssnr {

// ---------------------------------------------------------------------------
// Formats
// ---------------------------------------------------------------------------
VkFormat CompositionFormat(VkFormat swapchainFormat) {
    switch (swapchainFormat) {
        case VK_FORMAT_B8G8R8A8_UNORM:
        case VK_FORMAT_B8G8R8A8_SRGB:
            return VK_FORMAT_B8G8R8A8_UNORM;
        case VK_FORMAT_R8G8B8A8_UNORM:
        case VK_FORMAT_R8G8B8A8_SRGB:
            return VK_FORMAT_R8G8B8A8_UNORM;
        case VK_FORMAT_A8B8G8R8_UNORM_PACK32:
        case VK_FORMAT_A8B8G8R8_SRGB_PACK32:
            return VK_FORMAT_A8B8G8R8_UNORM_PACK32;
        case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
            return VK_FORMAT_A2B10G10R10_UNORM_PACK32;
        case VK_FORMAT_A2R10G10B10_UNORM_PACK32:
            return VK_FORMAT_A2R10G10B10_UNORM_PACK32;
        case VK_FORMAT_R16G16B16A16_SFLOAT:
            return VK_FORMAT_R16G16B16A16_SFLOAT;
        default:
            return VK_FORMAT_UNDEFINED;
    }
}

bool ColourIsLinearHdr(VkFormat swapchainFormat, uint32_t colourMode) {
    if (colourMode == kColourDisplay) return false;
    if (colourMode == kColourLinearHdr) return true;
    // Auto. An 8-bit frame has been tone mapped or there would be nothing to see, and HDR10's
    // ten-bit formats carry PQ, which is display-referred as well. Only a float swapchain is light.
    return swapchainFormat == VK_FORMAT_R16G16B16A16_SFLOAT;
}

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------
FrameSettings FrameSettings::Read(const ShmHeader* h) {
    FrameSettings s;
    if (!h) return s;
    s.transferStrength = BitsToFloat(h->transferStrengthBits.load());
    s.colourStrength = BitsToFloat(h->colourStrengthBits.load());
    s.maxRatio = BitsToFloat(h->maxRatioBits.load());
    s.debugScale = BitsToFloat(h->debugScaleBits.load());
    s.compareSplit = BitsToFloat(h->compareSplitBits.load());
    s.compareZoom = BitsToFloat(h->compareZoomBits.load());
    s.workingScale = BitsToFloat(h->workingScaleBits.load());
    s.transfer = h->transfer.load();
    s.debugView = h->debugView.load();
    s.compareMode = h->compareMode.load();
    s.compareSwap = h->compareSwap.load();
    s.reversibleMode = h->reversibleMode.load();
    s.applyModel = h->applyModel.load();
    s.holdFrame = h->holdFrame.load();

    // The slider, times the scale that says what the model should consider white. The trim belongs
    // to the measured source, which this layer does not have yet.
    const float base = BitsToFloat(h->whitePointBits.load());
    const float scale = BitsToFloat(h->whitePointScaleBits.load());
    s.whitePoint = base * (std::isfinite(scale) && scale > 0.0f ? scale : 1.0f);

    // Clamped here rather than trusted, because these come from a file any process can write.
    const auto clamp = [](float v, float lo, float hi, float fallback) {
        if (!std::isfinite(v)) return fallback;
        return std::min(std::max(v, lo), hi);
    };
    s.transferStrength = clamp(s.transferStrength, 0.0f, 4.0f, 1.0f);
    s.colourStrength = clamp(s.colourStrength, 0.0f, 4.0f, 1.0f);
    s.maxRatio = clamp(s.maxRatio, 1.0f, float(kMaxPasses), 2.0f);
    s.debugScale = clamp(s.debugScale, 0.01f, 100.0f, 1.0f);
    s.whitePoint = clamp(s.whitePoint, 1e-4f, 2000.0f, 1.0f);
    s.compareSplit = clamp(s.compareSplit, 0.0f, 1.0f, 0.5f);
    s.compareZoom = clamp(s.compareZoom, 1.0f, 2.0f, 1.0f);

    // Above 1.0 is supersampling, which needs a down-leg this layer has not ported yet; clamping
    // rather than refusing means the control exists and simply stops at 100% for now.
    s.workingScale = clamp(s.workingScale, 0.25f, 1.0f, 1.0f);

    if (s.transfer > 1) s.transfer = 1;
    if (s.debugView > 3) s.debugView = 0;
    if (s.compareMode > 2) s.compareMode = 0;
    if (s.reversibleMode >= kReversibleModeCount) s.reversibleMode = kReversibleKnee;
    return s;
}

// ---------------------------------------------------------------------------
// Lifetime
// ---------------------------------------------------------------------------
Composition::Composition(const DeviceTable* vk, const InstanceTable* instance, VkDevice device,
                         VkPhysicalDevice physicalDevice)
    : _vk(vk), _instance(instance), _device(device), _physicalDevice(physicalDevice) {
    if (!DeviceTableComplete(*vk)) {
        _reason = "the device does not expose everything a compute pass needs";
        Log("[comp] %s", _reason.c_str());
        return;
    }

    _pass = std::make_unique<DlssNrPass>(vk, instance, device, physicalDevice);
    if (!_pass->CanRender()) {
        _reason = "the composition pipeline could not be built";
        _pass.reset();
        return;
    }

    _usable = true;
}

Composition::~Composition() {
    DropAll();
    _pass.reset();
}

void Composition::DropAll() {
    DropImage(_frame);
    DropImage(_keep);
    DropImage(_proxy);
    DropImage(_work);
    DropImage(_model);
    DropImage(_composed);
    DropHostBuffer(_download);
    DropHostBuffer(_upload);
    _width = _height = _modelW = _modelH = 0;
    _haveModel = false;
}

bool Composition::FormatSupportsStorage(VkFormat format) const {
    VkFormatProperties props{};
    _instance->vkGetPhysicalDeviceFormatProperties(_physicalDevice, format, &props);
    return (props.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) != 0;
}

bool Composition::MakeImage(Image& img, uint32_t w, uint32_t h, VkFormat format, VkImageUsageFlags usage) {
    DropImage(img);

    VkImageCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = format;
    ci.extent = { w, h, 1 };
    ci.mipLevels = 1;
    ci.arrayLayers = 1;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = usage;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    if (_vk->vkCreateImage(_device, &ci, nullptr, &img.image) != VK_SUCCESS) {
        Log("[comp] vkCreateImage %ux%u fmt=%d failed", w, h, (int) format);
        return false;
    }

    VkMemoryRequirements req{};
    _vk->vkGetImageMemoryRequirements(_device, img.image, &req);

    VkPhysicalDeviceMemoryProperties mp{};
    _instance->vkGetPhysicalDeviceMemoryProperties(_physicalDevice, &mp);
    uint32_t type = UINT32_MAX;
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        if ((req.memoryTypeBits & (1u << i)) &&
            (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            type = i;
            break;
        }
    }
    if (type == UINT32_MAX) return false;

    VkMemoryAllocateInfo mai{};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = type;
    if (_vk->vkAllocateMemory(_device, &mai, nullptr, &img.memory) != VK_SUCCESS) {
        Log("[comp] out of device memory for a %ux%u surface", w, h);
        return false;
    }
    if (_vk->vkBindImageMemory(_device, img.image, img.memory, 0) != VK_SUCCESS) return false;

    VkImageViewCreateInfo vi{};
    vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vi.image = img.image;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = format;
    vi.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    if (_vk->vkCreateImageView(_device, &vi, nullptr, &img.view) != VK_SUCCESS) return false;

    img.format = format;
    img.width = w;
    img.height = h;
    img.layout = VK_IMAGE_LAYOUT_UNDEFINED;
    return true;
}

void Composition::DropImage(Image& img) {
    if (img.view) _vk->vkDestroyImageView(_device, img.view, nullptr);
    if (img.image) _vk->vkDestroyImage(_device, img.image, nullptr);
    if (img.memory) _vk->vkFreeMemory(_device, img.memory, nullptr);
    img = Image{};
}

bool Composition::MakeHostBuffer(HostBuffer& buf, size_t bytes, VkBufferUsageFlags usage) {
    DropHostBuffer(buf);

    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = bytes;
    bci.usage = usage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (_vk->vkCreateBuffer(_device, &bci, nullptr, &buf.buffer) != VK_SUCCESS) return false;

    VkMemoryRequirements req{};
    _vk->vkGetBufferMemoryRequirements(_device, buf.buffer, &req);

    VkPhysicalDeviceMemoryProperties mp{};
    _instance->vkGetPhysicalDeviceMemoryProperties(_physicalDevice, &mp);

    // Host visible and coherent is the requirement; cached is a large win on the readback and
    // harmless on the upload, so it is preferred rather than demanded.
    const VkMemoryPropertyFlags required =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    int best = -1, bestScore = -1;
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        if (!(req.memoryTypeBits & (1u << i))) continue;
        const VkMemoryPropertyFlags f = mp.memoryTypes[i].propertyFlags;
        if ((f & required) != required) continue;
        int score = 0;
        if (f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) score += 100;
        if (f & VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT) score -= 50;
        if (score > bestScore) { bestScore = score; best = (int) i; }
    }
    if (best < 0) return false;

    VkMemoryAllocateInfo mai{};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = uint32_t(best);
    if (_vk->vkAllocateMemory(_device, &mai, nullptr, &buf.memory) != VK_SUCCESS) return false;
    if (_vk->vkBindBufferMemory(_device, buf.buffer, buf.memory, 0) != VK_SUCCESS) return false;
    if (_vk->vkMapMemory(_device, buf.memory, 0, VK_WHOLE_SIZE, 0, &buf.mapped) != VK_SUCCESS) return false;

    buf.size = bytes;
    return true;
}

void Composition::DropHostBuffer(HostBuffer& buf) {
    if (buf.mapped) _vk->vkUnmapMemory(_device, buf.memory);
    if (buf.buffer) _vk->vkDestroyBuffer(_device, buf.buffer, nullptr);
    if (buf.memory) _vk->vkFreeMemory(_device, buf.memory, nullptr);
    buf = HostBuffer{};
}

// ---------------------------------------------------------------------------
// Sizing
// ---------------------------------------------------------------------------
bool Composition::Prepare(uint32_t width, uint32_t height, VkFormat swapchainFormat, const FrameSettings& s,
                          bool linearHdr) {
    if (!_usable) return false;

    const VkFormat work = CompositionFormat(swapchainFormat);
    if (work == VK_FORMAT_UNDEFINED) {
        _reason = "unsupported swapchain format";
        return false;
    }

    // The composed surface is written as a storage image and then copied, byte for byte, into the
    // swapchain. Both halves of that constrain it to the swapchain's own UNORM twin: a different
    // format would either be rejected by the copy or reinterpret the channels.
    if (!FormatSupportsStorage(work)) {
        _reason = "this device cannot write the swapchain's format as a storage image";
        Log("[comp] %s (format %d)", _reason.c_str(), (int) work);
        _usable = false;
        return false;
    }

    // A known divergence from the spec, carried deliberately and inherited from upstream.
    //
    // The vendored module declares its two storage images with an explicit format operand, Rgba32f.
    // Vulkan wants the bound image view's format to match that, and none of the views here do: the
    // proxy is R8G8B8A8_UNORM because eight bits is all that crosses to the model, and the composed
    // surface has to be the swapchain's own UNORM twin or the copy back would not be byte-exact.
    // Validation reports it as Undefined-Value-StorageImage-FormatMismatch-ImageView.
    //
    // Upstream has the same mismatch -- DlssNrFeature_Vk binds R16G16B16A16_SFLOAT to the same
    // Rgba32f declaration -- so this is the arrangement the shader has always run in. Measured on an
    // RTX 5090 by reading the composed surface back and comparing it against the frame it was made
    // from: means within a tenth of a level per channel, centre pixel within one level, alpha exact.
    // The writes land correctly on this driver.
    //
    // It is still undefined behaviour by the letter of the spec, and the two real fixes are known:
    // recompile the shader with [[vk::image_format]] matching what is bound, which cannot work while
    // one binding serves surfaces of three different formats; or recompile it with an Unknown format
    // and enable shaderStorageImageWriteWithoutFormat on the device, which is the arrangement this
    // pass actually needs and which vkBasalt reaches by a similar route.
    //
    // The model works at a fraction of the frame. Rounded to a multiple of eight so the dispatch
    // covers it exactly, and never below 64 so a tiny window cannot produce a degenerate raster.
    const auto scaled = [&](uint32_t v) {
        const uint32_t r = uint32_t(std::lround(double(v) * double(s.workingScale)));
        return std::max<uint32_t>(64, (r + 7u) & ~7u);
    };
    const uint32_t modelW = std::min(scaled(width), width);
    const uint32_t modelH = std::min(scaled(height), height);

    if (_width == width && _height == height && _swapchainFormat == swapchainFormat &&
        _modelW == modelW && _modelH == modelH && _linearHdr == linearHdr && _frame.image)
        return true;

    Log("[comp] building %ux%u, model %ux%u, %s", width, height, modelW, modelH,
        linearHdr ? "linear HDR" : "display-referred");

    DropAll();

    _swapchainFormat = swapchainFormat;
    _workFormat = work;
    _linearHdr = linearHdr;

    // The untouched copy is float only when the frame it holds is: on a display-referred frame the
    // swapchain's own format loses nothing and costs half the memory.
    _keepFormat = linearHdr ? VK_FORMAT_R16G16B16A16_SFLOAT : work;

    const VkImageUsageFlags sampled = VK_IMAGE_USAGE_SAMPLED_BIT;
    const VkImageUsageFlags storage = VK_IMAGE_USAGE_STORAGE_BIT;
    const VkImageUsageFlags src = VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    const VkImageUsageFlags dst = VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    const bool ok =
        MakeImage(_frame, width, height, work, sampled | dst) &&
        MakeImage(_keep, width, height, _keepFormat, sampled | storage) &&
        MakeImage(_proxy, width, height, VK_FORMAT_R8G8B8A8_UNORM, sampled | storage | src) &&
        MakeImage(_model, modelW, modelH, VK_FORMAT_R8G8B8A8_UNORM, sampled | dst) &&
        MakeImage(_composed, width, height, work, storage | src) &&
        MakeHostBuffer(_download, size_t(modelW) * modelH * 4, dst) &&
        MakeHostBuffer(_upload, size_t(modelW) * modelH * 4, src);

    const bool needWork = (modelW != width || modelH != height);
    const bool okWork = !needWork || MakeImage(_work, modelW, modelH, VK_FORMAT_R8G8B8A8_UNORM,
                                               sampled | storage | src);

    if (!ok || !okWork) {
        _reason = "could not allocate the composition surfaces";
        DropAll();
        return false;
    }

    _width = width;
    _height = height;
    _modelW = modelW;
    _modelH = modelH;
    _reason.clear();
    return true;
}

// ---------------------------------------------------------------------------
// Barriers
// ---------------------------------------------------------------------------
void Composition::Transition(VkCommandBuffer cb, Image& img, VkImageLayout to) {
    if (img.layout == to) return;
    VkImageSubresourceRange range{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    _pass->SetImageLayout(cb, img.image, img.layout, to, range);
    img.layout = to;
}

void Composition::TransitionSwapchain(VkCommandBuffer cb, VkImage image, VkImageLayout from, VkImageLayout to) {
    VkImageMemoryBarrier b{};
    b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b.oldLayout = from;
    b.newLayout = to;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image;
    b.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    b.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    _vk->vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0,
                              nullptr, 0, nullptr, 1, &b);
}

void Composition::CopyWholeImage(VkCommandBuffer cb, VkImage src, VkImageLayout srcLayout, VkImage dst,
                                 VkImageLayout dstLayout, uint32_t w, uint32_t h) {
    VkImageCopy copy{};
    copy.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    copy.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    copy.extent = { w, h, 1 };
    _vk->vkCmdCopyImage(cb, src, srcLayout, dst, dstLayout, 1, &copy);
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
DlssNrConstants Composition::BaseConstants(const FrameSettings& s) const {
    DlssNrConstants c{};
    c.WhitePoint = s.whitePoint;
    c.TransferStrength = s.transferStrength;
    c.ColourStrength = s.colourStrength;
    c.MaxRatio = s.maxRatio;
    c.DebugView = s.debugView;
    c.DebugScale = s.debugScale;
    c.Transfer = s.transfer;
    c.CompareMode = s.compareMode;
    c.CompareSplit = s.compareSplit;
    c.CompareZoom = s.compareZoom;
    c.CompareSwap = s.compareSwap;
    c.ReversibleMode = s.reversibleMode;
    c.ApplyModel = s.applyModel;

    // A frame the game already tone mapped goes through the encode untouched, and the composition
    // works in its units rather than normalising by a white point that means nothing here.
    c.Passthrough = _linearHdr ? 0u : 1u;

    // No motion vectors and no exposure reach a present-time layer. The guides are declared at the
    // frame's own size so nothing downstream scales by a ratio that does not exist.
    c.MvScaleX = 1.0f;
    c.MvScaleY = 1.0f;
    c.GuideWidth = _width;
    c.GuideHeight = _height;
    c.UseGameExposure = 0;
    c.ExposurePreMul = 1.0f;
    return c;
}

// ---------------------------------------------------------------------------
// Leg 1: the frame the model is shown
// ---------------------------------------------------------------------------
bool Composition::RecordCapture(VkCommandBuffer cb, VkImage swapchainImage, const FrameSettings& s) {
    if (!_usable || !_frame.image) return false;

    // Frame hold: keep working on the picture already captured, so a setting changed now is compared
    // against the same frame rather than against whatever the game has drawn since.
    if (s.holdFrame && _haveModel) return true;

    TransitionSwapchain(cb, swapchainImage, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    Transition(cb, _frame, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    CopyWholeImage(cb, swapchainImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, _frame.image,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, _width, _height);

    // Straight back, so that every path out of this leg -- including the ones that give up -- leaves
    // the image in the layout the present engine requires.
    TransitionSwapchain(cb, swapchainImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

    Transition(cb, _frame, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    Transition(cb, _proxy, VK_IMAGE_LAYOUT_GENERAL);
    Transition(cb, _keep, VK_IMAGE_LAYOUT_GENERAL);

    DlssNrConstants enc = BaseConstants(s);
    enc.Mode = DlssNrMode_Encode;
    enc.Width = _width;
    enc.Height = _height;
    if (!_pass->Dispatch(cb, enc, _width, _height, _frame.view, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE,
                         _proxy.view, _keep.view))
        return false;

    // What the model is actually handed: the full-resolution proxy, or a reduction of it.
    Image* source = &_proxy;
    if (_work.image) {
        Transition(cb, _proxy, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        Transition(cb, _work, VK_IMAGE_LAYOUT_GENERAL);

        DlssNrConstants down = BaseConstants(s);
        down.Mode = DlssNrMode_Downsample;
        down.Width = _modelW;
        down.Height = _modelH;
        if (!_pass->Dispatch(cb, down, _modelW, _modelH, _proxy.view, VK_NULL_HANDLE, VK_NULL_HANDLE,
                             VK_NULL_HANDLE, _work.view, VK_NULL_HANDLE))
            return false;
        source = &_work;
    }

    Transition(cb, *source, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

    VkBufferImageCopy region{};
    region.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    region.imageExtent = { _modelW, _modelH, 1 };
    _vk->vkCmdCopyImageToBuffer(cb, source->image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, _download.buffer, 1,
                                &region);
    return true;
}

// ---------------------------------------------------------------------------
// Leg 2: the model's answer, composed back
// ---------------------------------------------------------------------------
bool Composition::RecordCompose(VkCommandBuffer cb, VkImage swapchainImage, const FrameSettings& s) {
    if (!_usable || !_composed.image) return false;

    Transition(cb, _model, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    VkBufferImageCopy region{};
    region.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    region.imageExtent = { _modelW, _modelH, 1 };
    _vk->vkCmdCopyBufferToImage(cb, _upload.buffer, _model.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    Transition(cb, _model, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    Image* source = _work.image ? &_work : &_proxy;
    Transition(cb, *source, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    Transition(cb, _keep, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    Transition(cb, _composed, VK_IMAGE_LAYOUT_GENERAL);

    DlssNrConstants res = BaseConstants(s);
    res.Mode = DlssNrMode_Resolve;
    res.Width = _width;
    res.Height = _height;
    if (!_pass->Dispatch(cb, res, _width, _height, source->view, _model.view, _keep.view, VK_NULL_HANDLE,
                         _composed.view, VK_NULL_HANDLE))
        return false;

    Transition(cb, _composed, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    TransitionSwapchain(cb, swapchainImage, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    CopyWholeImage(cb, _composed.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, swapchainImage,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, _width, _height);
    TransitionSwapchain(cb, swapchainImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

    // The keep is written by the next encode, so it goes back to where that dispatch expects it.
    Transition(cb, _keep, VK_IMAGE_LAYOUT_GENERAL);
    return true;
}

}  // namespace dlssnr
