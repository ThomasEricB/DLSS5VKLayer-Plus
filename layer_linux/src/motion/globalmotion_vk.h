#pragma once
// How far the picture moved between the frame the model's answer belongs to and the frame being
// presented. See globalmotion.comp for why this is measured here rather than taken from the helper.
#include "shader_vk.h"

#include <memory>

namespace dlssnr {

// One of the three passes. Each is an ordinary Shader_Vk -- its own bindings, its own pipeline -- so
// nothing new had to be invented to run three dispatches instead of one.
class GmPass : public Shader_Vk {
  public:
    GmPass(const char* name, const DeviceTable* vk, const InstanceTable* instance, VkDevice device,
           VkPhysicalDevice physicalDevice, const unsigned char* spv, size_t spvBytes,
           const std::vector<VkDescriptorType>& bindings, size_t constantBytes, bool wantSampler);

    // Writes this dispatch's constants into the ring and returns the set to bind.
    VkDescriptorSet Begin(const void* constants, size_t bytes, VkDescriptorBufferInfo* uboOut);
    VkPipeline Pipeline() const { return _pipeline; }
    VkPipelineLayout Layout() const { return _pipelineLayout; }
    VkSampler Sampler() const { return _textureSampler; }
    const DeviceTable* Vk() const { return _vk; }
    VkDevice Device() const { return _device; }

  private:
    // Generous on purpose. The reduce pass runs four times a frame -- two levels, two pictures --
    // against the layer's three-deep command buffer ring, so a dozen of its sets can be pending at
    // once. Rewriting a set a submitted command buffer still points at is undefined behaviour, not a
    // stall, and this is the second time in this project that a ring sized by eye has been too small.
    static constexpr uint32_t kSlots = 32;
    VkDeviceSize _stride = 0;
    uint32_t _slot = 0;
};

class GlobalMotionVk {
  public:
    GlobalMotionVk(const DeviceTable* vk, const InstanceTable* instance, VkDevice device,
                   VkPhysicalDevice physicalDevice, uint32_t frameWidth, uint32_t frameHeight);
    ~GlobalMotionVk();

    bool CanRender() const { return _ok; }

    // Estimates the displacement from `then` to `now` and leaves it in a buffer the composition
    // reads. Both views must be the frame's own size; they are reduced internally.
    bool Record(VkCommandBuffer cb, VkImageView now, VkImageView then);

    // A one-by-one image holding (dx, dy, confidence, 1) in pixels of the frame. Sampled as a motion
    // field by the composition, where a single texel reads as one vector for the whole picture.
    VkImageView ResultView() const { return _result[1].view; }
    VkImage ResultImage() const { return _result[1].image; }

  private:
    struct Img {
        VkImage image = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
        VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
    };
    bool MakeImg(Img& img, uint32_t w, uint32_t h, VkFormat format = VK_FORMAT_R32_SFLOAT,
                 VkImageUsageFlags extra = 0);
    void DropImg(Img& img);
    void Barrier(VkCommandBuffer cb, Img& img, VkImageLayout to);

  public:
    // The composition reads the result, so it needs it in the right layout first.
    void BarrierResultForRead(VkCommandBuffer cb);

  private:

    const DeviceTable* _vk = nullptr;
    VkDevice _device = VK_NULL_HANDLE;
    VkPhysicalDevice _physical = VK_NULL_HANDLE;

    // Fixed working size, so the estimate costs the same on a 4K display as on a 720p one. Wide
    // enough that a displacement worth correcting is several cells across, small enough that a
    // few hundred candidate offsets over it is nothing.
    uint32_t _w[2]{}, _h[2]{};
    uint32_t _frameW = 0, _frameH = 0;
    int _radius[2] = { 12, 3 };

    // Two levels. The coarse one finds the displacement at all; the fine one pins it down to about a
    // pixel, which is what fine detail needs to stay correlated.
    Img _now[2]{}, _then[2]{}, _result[2]{};
    VkBuffer _costBuf = VK_NULL_HANDLE;
    VkDeviceMemory _costMem = VK_NULL_HANDLE;

    std::unique_ptr<GmPass> _reduce, _match, _pick;
    bool _ok = false;
};

}  // namespace dlssnr
