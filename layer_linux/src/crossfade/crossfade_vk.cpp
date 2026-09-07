#include "crossfade_vk.h"

#include "Crossfade_Shader_Vk.h"

#include <algorithm>
#include <cstring>

namespace dlssnr {

CrossfadeVk::CrossfadeVk(const DeviceTable* vk, const InstanceTable* instance, VkDevice device,
                         VkPhysicalDevice physicalDevice)
    : Shader_Vk("dlssnr-crossfade", vk, instance, device, physicalDevice) {
    if (device == VK_NULL_HANDLE) return;

    VkPhysicalDeviceProperties props{};
    _instance->vkGetPhysicalDeviceProperties(_physicalDevice, &props);
    const VkDeviceSize alignment = std::max<VkDeviceSize>(props.limits.minUniformBufferOffsetAlignment, 1);
    _slotStride = ((sizeof(Constants) + alignment - 1) / alignment) * alignment;

    if (!CreateBufferResource(&_constantBuffer, &_constantBufferMemory, _slotStride * kSlots,
                              VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
        return;
    if (_vk->vkMapMemory(_device, _constantBufferMemory, 0, _slotStride * kSlots, 0, &_mappedConstantBuffer) !=
        VK_SUCCESS)
        return;

    std::vector<VkDescriptorSetLayoutBinding> bindings = {
        CreateBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER),
        CreateBinding(1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE),
        CreateBinding(2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE),
    };
    CreateLayouts(bindings);
    if (_descriptorSetLayout == VK_NULL_HANDLE || _pipelineLayout == VK_NULL_HANDLE) return;

    std::vector<VkDescriptorPoolSize> poolSizes = {
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kSlots },
        { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kSlots },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kSlots },
    };
    CreateDescriptorPool(poolSizes, kSlots);
    CreateDescriptorSets(kSlots);
    if (_descriptorSets.size() < kSlots) return;

    const std::vector<char> code((const char*) crossfade_spv,
                                 (const char*) crossfade_spv + sizeof(crossfade_spv));
    if (!CreateComputePipeline(_pipelineLayout, &_pipeline, code)) return;

    _init = true;
}

bool CrossfadeVk::Dispatch(VkCommandBuffer cb, VkImageView target, VkImageView current, uint32_t width,
                           uint32_t height, float alpha) {
    if (!CanRender() || cb == VK_NULL_HANDLE || target == VK_NULL_HANDLE || current == VK_NULL_HANDLE)
        return false;

    const uint32_t slot = _slot;
    _slot = (_slot + 1) % kSlots;
    const VkDeviceSize offset = _slotStride * slot;

    Constants c{};
    c.width = width;
    c.height = height;
    c.alpha = alpha;
    std::memcpy((char*) _mappedConstantBuffer + offset, &c, sizeof(c));

    VkDescriptorSet set = _descriptorSets[slot];
    VkDescriptorBufferInfo bufferInfo{ _constantBuffer, offset, sizeof(Constants) };
    VkDescriptorImageInfo targetInfo{ VK_NULL_HANDLE, target, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    VkDescriptorImageInfo currentInfo{ VK_NULL_HANDLE, current, VK_IMAGE_LAYOUT_GENERAL };

    const VkWriteDescriptorSet writes[] = {
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 0, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
          nullptr, &bufferInfo, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 1, 0, 1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
          &targetInfo, nullptr, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
          &currentInfo, nullptr, nullptr },
    };
    _vk->vkUpdateDescriptorSets(_device, uint32_t(sizeof(writes) / sizeof(writes[0])), writes, 0, nullptr);

    _vk->vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, _pipeline);
    _vk->vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, _pipelineLayout, 0, 1, &set, 0, nullptr);
    _vk->vkCmdDispatch(cb, (width + 7) / 8, (height + 7) / 8, 1);

    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    _vk->vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1,
                              &barrier, 0, nullptr, 0, nullptr);
    return true;
}

}  // namespace dlssnr
