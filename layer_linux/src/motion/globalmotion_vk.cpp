#include "globalmotion_vk.h"

#include "GlobalMotion_Shaders.h"
#include "log.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace dlssnr {

GmPass::GmPass(const char* name, const DeviceTable* vk, const InstanceTable* instance, VkDevice device,
               VkPhysicalDevice physicalDevice, const unsigned char* spv, size_t spvBytes,
               const std::vector<VkDescriptorType>& bindings, size_t constantBytes, bool wantSampler)
    : Shader_Vk(name, vk, instance, device, physicalDevice) {
    if (device == VK_NULL_HANDLE) return;

    if (wantSampler) {
        CreateSampler(VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
        if (_textureSampler == VK_NULL_HANDLE) return;
    }

    VkPhysicalDeviceProperties props{};
    _instance->vkGetPhysicalDeviceProperties(_physicalDevice, &props);
    const VkDeviceSize align = std::max<VkDeviceSize>(props.limits.minUniformBufferOffsetAlignment, 1);
    _stride = ((constantBytes + align - 1) / align) * align;

    if (!CreateBufferResource(&_constantBuffer, &_constantBufferMemory, _stride * kSlots,
                              VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
        return;
    if (_vk->vkMapMemory(_device, _constantBufferMemory, 0, _stride * kSlots, 0, &_mappedConstantBuffer) !=
        VK_SUCCESS)
        return;

    std::vector<VkDescriptorSetLayoutBinding> binds;
    std::vector<VkDescriptorPoolSize> sizes;
    for (uint32_t i = 0; i < bindings.size(); ++i) {
        binds.push_back(CreateBinding(i, bindings[i]));
        sizes.push_back({ bindings[i], kSlots });
    }
    CreateLayouts(binds);
    if (_descriptorSetLayout == VK_NULL_HANDLE || _pipelineLayout == VK_NULL_HANDLE) return;
    CreateDescriptorPool(sizes, kSlots);
    CreateDescriptorSets(kSlots);
    if (_descriptorSets.size() < kSlots) return;

    const std::vector<char> code((const char*) spv, (const char*) spv + spvBytes);
    // "main", not "CSMain": these are GLSL, where the entry point is spelled the other way.
    if (!CreateComputePipeline(_pipelineLayout, &_pipeline, code, "main")) return;

    _init = true;
}

VkDescriptorSet GmPass::Begin(const void* constants, size_t bytes, VkDescriptorBufferInfo* uboOut) {
    const uint32_t slot = _slot;
    _slot = (_slot + 1) % kSlots;
    const VkDeviceSize offset = _stride * slot;
    std::memcpy((char*) _mappedConstantBuffer + offset, constants, bytes);
    *uboOut = VkDescriptorBufferInfo{ _constantBuffer, offset, bytes };
    return _descriptorSets[slot];
}

// ---------------------------------------------------------------------------

GlobalMotionVk::GlobalMotionVk(const DeviceTable* vk, const InstanceTable* instance, VkDevice device,
                               VkPhysicalDevice physicalDevice, uint32_t frameWidth, uint32_t frameHeight)
    : _vk(vk), _device(device), _physical(physicalDevice), _frameW(frameWidth), _frameH(frameHeight) {
    if (!vk || device == VK_NULL_HANDLE || !frameWidth || !frameHeight) return;

    // A fixed working width, with the frame's aspect. The estimate then costs the same whatever the
    // display is, and the search radius means the same fraction of the picture everywhere.
    for (int lvl = 0; lvl < 2; ++lvl) {
        _w[lvl] = 160u << lvl;
        _h[lvl] = std::max<uint32_t>(16, uint32_t(std::lround(double(_w[lvl]) * double(frameHeight) /
                                                              double(frameWidth))));
        if (!MakeImg(_now[lvl], _w[lvl], _h[lvl]) || !MakeImg(_then[lvl], _w[lvl], _h[lvl]) ||
            !MakeImg(_result[lvl], 1, 1, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT))
            return;
    }

    const int span = 2 * std::max(_radius[0], _radius[1]) + 1;
    auto makeBuf = [&](VkBuffer* b, VkDeviceMemory* m, VkDeviceSize size) {
        VkBufferCreateInfo bi{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        bi.size = size;
        bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (_vk->vkCreateBuffer(_device, &bi, nullptr, b) != VK_SUCCESS) return false;
        VkMemoryRequirements mr{};
        _vk->vkGetBufferMemoryRequirements(_device, *b, &mr);
        VkPhysicalDeviceMemoryProperties mp{};
        instance->vkGetPhysicalDeviceMemoryProperties(_physical, &mp);
        uint32_t type = UINT32_MAX;
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
            if ((mr.memoryTypeBits & (1u << i)) &&
                (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) { type = i; break; }
        if (type == UINT32_MAX) return false;
        VkMemoryAllocateInfo ai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
        ai.allocationSize = mr.size;
        ai.memoryTypeIndex = type;
        if (_vk->vkAllocateMemory(_device, &ai, nullptr, m) != VK_SUCCESS) return false;
        return _vk->vkBindBufferMemory(_device, *b, *m, 0) == VK_SUCCESS;
    };
    if (!makeBuf(&_costBuf, &_costMem, VkDeviceSize(span) * span * sizeof(float))) return;

    _reduce = std::make_unique<GmPass>(
        "dlssnr-gm-reduce", vk, instance, device, physicalDevice, gm_reduce_spv, sizeof(gm_reduce_spv),
        std::vector<VkDescriptorType>{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                       VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                       VK_DESCRIPTOR_TYPE_STORAGE_IMAGE },
        4 * sizeof(uint32_t), true);
    _match = std::make_unique<GmPass>(
        "dlssnr-gm-match", vk, instance, device, physicalDevice, gm_match_spv, sizeof(gm_match_spv),
        std::vector<VkDescriptorType>{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                       VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                       VK_DESCRIPTOR_TYPE_STORAGE_IMAGE },
        8 * sizeof(uint32_t), false);
    _pick = std::make_unique<GmPass>(
        "dlssnr-gm-pick", vk, instance, device, physicalDevice, gm_pick_spv, sizeof(gm_pick_spv),
        std::vector<VkDescriptorType>{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                       VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE },
        4 * sizeof(uint32_t), false);

    _ok = _reduce->CanRender() && _match->CanRender() && _pick->CanRender();
    if (_ok)
        Log("[gm] global motion estimate ready, %ux%u then %ux%u, search +-%d then +-%d",
            _w[0], _h[0], _w[1], _h[1], _radius[0], _radius[1]);
}

GlobalMotionVk::~GlobalMotionVk() {
    _reduce.reset();
    _match.reset();
    _pick.reset();
    for (int lvl = 0; lvl < 2; ++lvl) {
        DropImg(_now[lvl]);
        DropImg(_then[lvl]);
        DropImg(_result[lvl]);
    }
    if (_costBuf) _vk->vkDestroyBuffer(_device, _costBuf, nullptr);
    if (_costMem) _vk->vkFreeMemory(_device, _costMem, nullptr);
}

bool GlobalMotionVk::MakeImg(Img& img, uint32_t w, uint32_t h, VkFormat format,
                             VkImageUsageFlags extra) {
    VkImageCreateInfo ci{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = format;
    ci.extent = { w, h, 1 };
    ci.mipLevels = 1;
    ci.arrayLayers = 1;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = VK_IMAGE_USAGE_STORAGE_BIT | extra;
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (_vk->vkCreateImage(_device, &ci, nullptr, &img.image) != VK_SUCCESS) return false;
    VkMemoryRequirements mr{};
    _vk->vkGetImageMemoryRequirements(_device, img.image, &mr);
    VkPhysicalDeviceMemoryProperties mp{};
    // The instance table is not kept, so the memory type is found from the requirements alone:
    // device-local is the only property wanted and every heap bit here satisfies it in practice.
    VkMemoryAllocateInfo ai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    ai.allocationSize = mr.size;
    ai.memoryTypeIndex = 0;
    for (uint32_t i = 0; i < 32; ++i)
        if (mr.memoryTypeBits & (1u << i)) { ai.memoryTypeIndex = i; break; }
    (void) mp;
    if (_vk->vkAllocateMemory(_device, &ai, nullptr, &img.memory) != VK_SUCCESS) return false;
    if (_vk->vkBindImageMemory(_device, img.image, img.memory, 0) != VK_SUCCESS) return false;
    VkImageViewCreateInfo vi{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    vi.image = img.image;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = ci.format;
    vi.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    return _vk->vkCreateImageView(_device, &vi, nullptr, &img.view) == VK_SUCCESS;
}

void GlobalMotionVk::DropImg(Img& img) {
    if (img.view) _vk->vkDestroyImageView(_device, img.view, nullptr);
    if (img.image) _vk->vkDestroyImage(_device, img.image, nullptr);
    if (img.memory) _vk->vkFreeMemory(_device, img.memory, nullptr);
    img = Img{};
}

void GlobalMotionVk::Barrier(VkCommandBuffer cb, Img& img, VkImageLayout to) {
    if (img.layout == to) return;
    VkImageMemoryBarrier b{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
    b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
    b.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
    b.oldLayout = img.layout;
    b.newLayout = to;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = img.image;
    b.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    _vk->vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              0, 0, nullptr, 0, nullptr, 1, &b);
    img.layout = to;
}

bool GlobalMotionVk::Record(VkCommandBuffer cb, VkImageView now, VkImageView then) {
    if (!_ok || now == VK_NULL_HANDLE || then == VK_NULL_HANDLE) return false;

    const auto write = [&](VkDescriptorSet set, uint32_t binding, VkDescriptorType type,
                           const VkDescriptorImageInfo* img, const VkDescriptorBufferInfo* buf) {
        VkWriteDescriptorSet w{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        w.dstSet = set;
        w.dstBinding = binding;
        w.descriptorCount = 1;
        w.descriptorType = type;
        w.pImageInfo = img;
        w.pBufferInfo = buf;
        _vk->vkUpdateDescriptorSets(_device, 1, &w, 0, nullptr);
    };
    VkMemoryBarrier mb{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    const auto flush = [&] {
        _vk->vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    };

    // Coarse first, to find the displacement at all; then fine, searching a small window around it.
    for (int lvl = 0; lvl < 2; ++lvl) {
        const uint32_t w = _w[lvl], h = _h[lvl];
        const int radius = _radius[lvl];
        const int span = 2 * radius + 1;
        const float cellX = float(_frameW) / float(w);
        const float cellY = float(_frameH) / float(h);
        const uint32_t useBase = lvl == 0 ? 0u : 1u;

        // --- reduce both frames to this level's size ---
        struct RC { uint32_t w, h, p0, p1; } rc{ w, h, 0, 0 };
        const auto reduceOne = [&](VkImageView src, Img& dst) {
            Barrier(cb, dst, VK_IMAGE_LAYOUT_GENERAL);
            VkDescriptorBufferInfo ubo{};
            VkDescriptorSet set = _reduce->Begin(&rc, sizeof(rc), &ubo);
            VkDescriptorImageInfo si{ _reduce->Sampler(), src, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
            VkDescriptorImageInfo di{ VK_NULL_HANDLE, dst.view, VK_IMAGE_LAYOUT_GENERAL };
            write(set, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &ubo);
            write(set, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &si, nullptr);
            write(set, 2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &di, nullptr);
            _vk->vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, _reduce->Pipeline());
            _vk->vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, _reduce->Layout(), 0, 1, &set,
                                         0, nullptr);
            _vk->vkCmdDispatch(cb, (w + 15) / 16, (h + 15) / 16, 1);
        };
        reduceOne(now, _now[lvl]);
        reduceOne(then, _then[lvl]);
        flush();

        // The base this level searches around: level 0 searches around nothing, level 1 around what
        // level 0 found. Bound either way, because a descriptor the shader may not read still has to
        // be there.
        Img& base = _result[lvl == 0 ? 1 : 0];
        Barrier(cb, base, VK_IMAGE_LAYOUT_GENERAL);
        Barrier(cb, _result[lvl], VK_IMAGE_LAYOUT_GENERAL);

        // --- score every candidate offset ---
        struct MC { uint32_t w, h; int radius; uint32_t useBase; float cellX, cellY; int p0, p1; } mc{
            w, h, radius, useBase, cellX, cellY, 0, 0
        };
        {
            VkDescriptorBufferInfo ubo{};
            VkDescriptorSet set = _match->Begin(&mc, sizeof(mc), &ubo);
            VkDescriptorImageInfo ni{ VK_NULL_HANDLE, _now[lvl].view, VK_IMAGE_LAYOUT_GENERAL };
            VkDescriptorImageInfo ti{ VK_NULL_HANDLE, _then[lvl].view, VK_IMAGE_LAYOUT_GENERAL };
            VkDescriptorImageInfo bi{ VK_NULL_HANDLE, base.view, VK_IMAGE_LAYOUT_GENERAL };
            VkDescriptorBufferInfo co{ _costBuf, 0, VK_WHOLE_SIZE };
            write(set, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &ubo);
            write(set, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &ni, nullptr);
            write(set, 2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &ti, nullptr);
            write(set, 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &co);
            write(set, 4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &bi, nullptr);
            _vk->vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, _match->Pipeline());
            _vk->vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, _match->Layout(), 0, 1, &set,
                                         0, nullptr);
            _vk->vkCmdDispatch(cb, uint32_t(span * span), 1, 1);
        }
        flush();

        // --- pick the winner, in pixels of the frame ---
        struct PC { int radius; float scaleX, scaleY; uint32_t useBase; } pc{ radius, cellX, cellY, useBase };
        {
            VkDescriptorBufferInfo ubo{};
            VkDescriptorSet set = _pick->Begin(&pc, sizeof(pc), &ubo);
            VkDescriptorBufferInfo ci{ _costBuf, 0, VK_WHOLE_SIZE };
            VkDescriptorImageInfo ri{ VK_NULL_HANDLE, _result[lvl].view, VK_IMAGE_LAYOUT_GENERAL };
            VkDescriptorImageInfo bi{ VK_NULL_HANDLE, base.view, VK_IMAGE_LAYOUT_GENERAL };
            write(set, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &ubo);
            write(set, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &ci);
            write(set, 2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &ri, nullptr);
            write(set, 3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &bi, nullptr);
            _vk->vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, _pick->Pipeline());
            _vk->vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, _pick->Layout(), 0, 1, &set, 0,
                                         nullptr);
            _vk->vkCmdDispatch(cb, 1, 1, 1);
        }
        flush();
    }
    return true;
}

void GlobalMotionVk::BarrierResultForRead(VkCommandBuffer cb) {
    if (_ok) Barrier(cb, _result[1], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

}  // namespace dlssnr
