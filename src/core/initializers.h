#pragma once

#include "common/vk_common.h"
#include "types.h"

namespace vkinit
{

[[nodiscard]]
vk::BufferCreateInfo bufferCreateInfo(vk::DeviceSize size,
                                    vk::BufferUsageFlags usage,
                                    const QueueFamilyIndices& indices);

[[nodiscard]]
vk::ImageCreateInfo imageCreateInfo(vk::Format format,
                                    vk::Extent3D extent,
                                    uint32_t mipLevels,
                                    vk::SampleCountFlagBits samples,
                                    vk::ImageTiling tiling,
                                    vk::ImageUsageFlags usage);

[[nodiscard]]
vk::ImageViewCreateInfo imageViewCreateInfo(vk::Image image,
                                            vk::Format format,
                                            vk::ImageAspectFlags aspectFlags,
                                            uint32_t mipLevels);

[[nodiscard]]
VmaAllocationCreateInfo vmaAllocationCreateInfo(MemoryType memoryType);

}   // namespace vkinit