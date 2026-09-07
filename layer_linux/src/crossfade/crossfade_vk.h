#pragma once
// The pass that softens a step into a ramp. See crossfade.hlsl for why blending the two terms
// separately is the same as blending the edit.
#include "shader_vk.h"

namespace dlssnr {

class CrossfadeVk : public Shader_Vk {
  public:
    CrossfadeVk(const DeviceTable* vk, const InstanceTable* instance, VkDevice device,
                VkPhysicalDevice physicalDevice);

    // Moves `current` a fraction alpha of the way toward `target`. Same size, no resampling.
    bool Dispatch(VkCommandBuffer cb, VkImageView target, VkImageView current, uint32_t width,
                  uint32_t height, float alpha);

  private:
    struct Constants {
        uint32_t width;
        uint32_t height;
        float alpha;
        uint32_t pad;
    };
    // Three dispatches a frame at most -- the answer, the proxy, and the working raster when the
    // model is not at the frame's size -- against the layer's three-deep command buffer ring, so nine
    // sets can be in the pending state at once. A set rewritten while a submitted command buffer
    // still refers to it is not a stall, it is undefined behaviour; sixteen leaves headroom.
    static constexpr uint32_t kSlots = 16;
    VkDeviceSize _slotStride = 0;
    uint32_t _slot = 0;
};

}  // namespace dlssnr
