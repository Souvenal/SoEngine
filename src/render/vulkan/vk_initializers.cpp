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

vk::RenderingAttachmentInfo colorAttachmentInfo(
        vk::ImageView imageView,
        vk::ImageLayout imageLayout,
        std::optional<vk::ClearValue> clearValue) {
    return vk::RenderingAttachmentInfo{
        .imageView = imageView,
        .imageLayout = imageLayout,
        .loadOp = clearValue ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = clearValue.value_or(vk::ClearValue{})};
}

vk::RenderingAttachmentInfo depthAttachmentInfo(
        vk::ImageView imageView,
        vk::ImageLayout layout) {
    return vk::RenderingAttachmentInfo{
        .imageView = imageView,
        .imageLayout = layout,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eDontCare,
        .clearValue = vk::ClearDepthStencilValue{1.f, 0}
    };
}

vk::RenderingInfo renderingInfo(
        vk::Extent2D extent,
        vk::RenderingAttachmentInfo colorAttachment,
        vk::RenderingAttachmentInfo depthAttachment) {
    return vk::RenderingInfo{
        .renderArea = { .offset = { 0, 0 }, .extent = extent },
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &colorAttachment,
        .pDepthAttachment = &depthAttachment,
        .pStencilAttachment = nullptr};
}

//> init submit
vk::CommandBufferSubmitInfo commandBufferSubmitInfo(
        const vk::CommandBuffer& commandBuffer) {
    return vk::CommandBufferSubmitInfo{
        .commandBuffer = commandBuffer,
        .deviceMask = 0};
}

vk::SemaphoreSubmitInfo semaphoreSubmitInfo(
        vk::PipelineStageFlags2 stageMask,
        vk::Semaphore semaphore,
        uint64_t value) {
    return vk::SemaphoreSubmitInfo{
        .semaphore = semaphore,
        .value = value,
        .stageMask = stageMask,
        .deviceIndex = 0};
}

vk::SubmitInfo2 submitInfo(
        const vk::CommandBufferSubmitInfo& commandBufferInfo,
        const vk::SemaphoreSubmitInfo& waitSemaphoreInfo,
        const vk::SemaphoreSubmitInfo& signalSemaphoreInfo) {
    return submitInfo(
        std::span{ &commandBufferInfo, 1 },
        std::span{ &waitSemaphoreInfo, 1 },
        std::span{ &signalSemaphoreInfo, 1 });
}

vk::SubmitInfo2 submitInfo(
        std::span<const vk::CommandBufferSubmitInfo> commandBufferInfos,
        std::span<const vk::SemaphoreSubmitInfo> waitSemaphoreInfos,
        std::span<const vk::SemaphoreSubmitInfo> signalSemaphoreInfos) {
    return vk::SubmitInfo2{
        .waitSemaphoreInfoCount = static_cast<uint32_t>(waitSemaphoreInfos.size()),
        .pWaitSemaphoreInfos = waitSemaphoreInfos.data(),
        .commandBufferInfoCount = static_cast<uint32_t>(commandBufferInfos.size()),
        .pCommandBufferInfos = commandBufferInfos.data(),
        .signalSemaphoreInfoCount = static_cast<uint32_t>(signalSemaphoreInfos.size()),
        .pSignalSemaphoreInfos = signalSemaphoreInfos.data()};
}
//< init submit

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