#pragma once

#include "common/vk_common.h"
#include "vk_types.h"

namespace vkinit
{

[[nodiscard]] vk::BufferCreateInfo bufferCreateInfo(
        vk::DeviceSize size,
        vk::BufferUsageFlags usage,
        const QueueFamilyIndices& indices);

[[nodiscard]] vk::ImageCreateInfo imageCreateInfo(
        vk::Format format,
        vk::Extent3D extent,
        uint32_t mipLevels,
        vk::SampleCountFlagBits samples,
        vk::ImageTiling tiling,
        vk::ImageUsageFlags usage);

[[nodiscard]] vk::ImageViewCreateInfo imageViewCreateInfo(
        vk::Image image,
        vk::Format format,
        vk::ImageAspectFlags aspectFlags,
        uint32_t mipLevels);

/**
 * @param clearValue If provided, the loadOp will be set to Clear, otherwise to Load.
 */
[[nodiscard]] vk::RenderingAttachmentInfo colorAttachmentInfo(
    vk::ImageView imageView,
    vk::ImageLayout imageLayout,
    std::optional<vk::ClearValue> clearValue = std::nullopt);

[[nodiscard]] vk::RenderingAttachmentInfo depthAttachmentInfo(
        vk::ImageView imageView,
        vk::ImageLayout imageLayout);

[[nodiscard]] vk::RenderingInfo renderingInfo(
        vk::Extent2D extent,
        vk::RenderingAttachmentInfo colorAttachment,
        vk::RenderingAttachmentInfo depthAttachment = {});

//> init submit
[[nodiscard]] vk::CommandBufferSubmitInfo commandBufferSubmitInfo(
        const vk::CommandBuffer& commandBuffer);

/**
 * @param value For timeline semaphores, the value to signal or wait on.
 */
[[nodiscard]] vk::SemaphoreSubmitInfo semaphoreSubmitInfo(
        vk::PipelineStageFlags2 stageMask,
        vk::Semaphore semaphore,
        uint64_t value = 1);

[[nodiscard]] vk::SubmitInfo2 submitInfo(
        const vk::CommandBufferSubmitInfo& commandBufferInfo,
        const vk::SemaphoreSubmitInfo& waitSemaphoreInfo,
        const vk::SemaphoreSubmitInfo& signalSemaphoreInfo);

[[nodiscard]] vk::SubmitInfo2 submitInfo(
        std::span<const vk::CommandBufferSubmitInfo> commandBufferInfos,
        std::span<const vk::SemaphoreSubmitInfo> waitSemaphoreInfos,
        std::span<const vk::SemaphoreSubmitInfo> signalSemaphoreInfos);
//< init submit

[[nodiscard]]
VmaAllocationCreateInfo vmaAllocationCreateInfo(MemoryType memoryType);

}   // namespace vkinit