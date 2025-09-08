#include "vk_initializers.h"

namespace vkinit
{

vk::BufferCreateInfo bufferCreateInfo(vk::DeviceSize size,
                                      vk::BufferUsageFlags usage,
                                      const QueueFamilyIndices& indices)
{
    uint32_t queueFamilyIndices[] = {
        indices.graphicsFamily.value(), indices.transferFamily.value()
    };
    
    vk::BufferCreateInfo bufferCreateInfo {
        .size = size,
        .usage = usage
    };
    if (indices.graphicsFamily.value() == indices.transferFamily.value()) {
        bufferCreateInfo.sharingMode = vk::SharingMode::eExclusive;
        bufferCreateInfo.queueFamilyIndexCount = 0;
        bufferCreateInfo.pQueueFamilyIndices = nullptr;
    } else {
        bufferCreateInfo.sharingMode = vk::SharingMode::eConcurrent;
        bufferCreateInfo.queueFamilyIndexCount = 2;
        bufferCreateInfo.pQueueFamilyIndices = queueFamilyIndices;
    }
    return bufferCreateInfo;
}

vk::ImageCreateInfo imageCreateInfo(vk::Format format,
                                    vk::Extent3D extent,
                                    uint32_t mipLevels,
                                    vk::SampleCountFlagBits samples,
                                    vk::ImageTiling tiling,
                                    vk::ImageUsageFlags usage)
{
    return vk::ImageCreateInfo{
        .imageType = vk::ImageType::e2D,
        .format = format,
        .extent = extent,
        .mipLevels = mipLevels,
        .arrayLayers = 1,
        .samples = samples,
        .tiling = tiling,
        .usage = usage,
        .sharingMode = vk::SharingMode::eExclusive,
        .initialLayout = vk::ImageLayout::eUndefined
    };
}

vk::ImageViewCreateInfo imageViewCreateInfo(vk::Image image,
                                            vk::Format format,
                                            vk::ImageAspectFlags aspectFlags,
                                            uint32_t mipLevels)
{
    return vk::ImageViewCreateInfo{
        .image = image,
        .viewType = vk::ImageViewType::e2D,
        .format = format,
        .components = {
            .r = vk::ComponentSwizzle::eIdentity,
            .g = vk::ComponentSwizzle::eIdentity,
            .b = vk::ComponentSwizzle::eIdentity,
            .a = vk::ComponentSwizzle::eIdentity
        },
        .subresourceRange = { aspectFlags, 0, mipLevels, 0, 1 }
    };
}

[[nodiscard]]
VmaAllocationCreateInfo vmaAllocationCreateInfo(MemoryType memoryType) {
    VmaAllocationCreateInfo vmaAllocCreateInfo {};
    if (memoryType == MemoryType::DeviceLocal) {
        vmaAllocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    } else if (memoryType == MemoryType::HostVisible) {
        vmaAllocCreateInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
        vmaAllocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
    } else {
        throw std::runtime_error("Unsupported BufferMemoryType");
    }
    return vmaAllocCreateInfo;
}

}   // namespace vkinit