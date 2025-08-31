#include "images.h"

#include <stdexcept>

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

vk::raii::ImageView createImageView(const vk::raii::Device& device,
                                        vk::Image image,
                                        vk::Format format,
                                        vk::ImageAspectFlags aspectFlags,
                                        uint32_t mipLevels)
{
    auto viewInfo = imageViewCreateInfo(image, format, aspectFlags, mipLevels);
    return device.createImageView(viewInfo);
}

AllocatedImage createAllocatedImage(const vk::raii::Device& device,
                                    VmaAllocator allocator,
                                    vk::Extent3D extent,
                                    uint32_t mipLevels,
                                    vk::SampleCountFlagBits numSamples,
                                    vk::Format format,
                                    vk::ImageTiling tiling,
                                    vk::ImageUsageFlags usage,
                                    vk::MemoryPropertyFlags properties)
{
    AllocatedImage allocatedImage {};
    allocatedImage.imageExtent = extent;
    allocatedImage.imageFormat = format;

    auto imageInfo = VkImageCreateInfo(imageCreateInfo(format, extent, mipLevels, numSamples, tiling, usage));
    VmaAllocationCreateInfo allocInfo {
        .usage = VMA_MEMORY_USAGE_AUTO,
        .requiredFlags = static_cast<VkMemoryPropertyFlags>(properties),
    };

    VkImage _image;
    if (vmaCreateImage(allocator, &imageInfo, &allocInfo,
        &_image, &allocatedImage.allocation, nullptr) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create image");
    }
    allocatedImage.image = _image;

    auto imageViewInfo = imageViewCreateInfo(allocatedImage.image, format, vk::ImageAspectFlagBits::eColor, mipLevels);

    allocatedImage.imageView = (*device).createImageView(imageViewInfo);

    return allocatedImage;
}

void transitionImageLayout(const vk::raii::CommandBuffer& commandBuffer,
                           vk::Image image,
                           vk::ImageLayout oldLayout,
                           vk::ImageLayout newLayout)
{
    vk::Flags<vk::ImageAspectFlagBits> aspectMask;
    if (newLayout == vk::ImageLayout::eDepthStencilAttachmentOptimal)
        aspectMask = vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil;
    else if (newLayout == vk::ImageLayout::eDepthAttachmentOptimal)
        aspectMask = vk::ImageAspectFlagBits::eDepth;
    else if (newLayout == vk::ImageLayout::eStencilAttachmentOptimal)
        aspectMask = vk::ImageAspectFlagBits::eStencil;
    else
        aspectMask = vk::ImageAspectFlagBits::eColor;

    vk::ImageSubresourceRange subresourceRange{
        aspectMask,
        0, 1, // baseMipLevel, levelCount
        0, 1  // baseArrayLayer, layerCount
    };

    transitionImageLayout(commandBuffer, image, oldLayout, newLayout, subresourceRange);
}

void transitionImageLayout(const vk::raii::CommandBuffer& commandBuffer,
                           vk::Image image,
                           vk::ImageLayout oldLayout,
                           vk::ImageLayout newLayout,
                           const vk::ImageSubresourceRange& subresourceRange)
{
    vk::PipelineStageFlagBits2 srcStageMask;
    vk::PipelineStageFlagBits2 dstStageMask;
    vk::Flags<vk::AccessFlagBits2> srcAccessMask;
    vk::Flags<vk::AccessFlagBits2> dstAccessMask;

    if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eTransferDstOptimal) {
        srcStageMask = vk::PipelineStageFlagBits2::eTopOfPipe;
        dstStageMask = vk::PipelineStageFlagBits2::eTransfer;
        srcAccessMask = {};
        dstAccessMask = vk::AccessFlagBits2::eTransferWrite;
    } else if (oldLayout == vk::ImageLayout::eTransferDstOptimal && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal) {
        srcStageMask = vk::PipelineStageFlagBits2::eTransfer;
        dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader;
        srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
        dstAccessMask = vk::AccessFlagBits2::eShaderRead;
    } else if (oldLayout == vk::ImageLayout::eUndefined &&
               (newLayout == vk::ImageLayout::eDepthStencilAttachmentOptimal ||
                newLayout == vk::ImageLayout::eDepthAttachmentOptimal)) {
        srcStageMask = vk::PipelineStageFlagBits2::eTopOfPipe;
        dstStageMask = vk::PipelineStageFlagBits2::eEarlyFragmentTests;
        srcAccessMask = {};
        dstAccessMask = vk::AccessFlagBits2::eDepthStencilAttachmentRead | vk::AccessFlagBits2::eDepthStencilAttachmentWrite;
    } else if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eColorAttachmentOptimal) {
        srcStageMask = vk::PipelineStageFlagBits2::eTopOfPipe;
        dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput;
        srcAccessMask = {};
        dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite;
    } else if (oldLayout == vk::ImageLayout::eColorAttachmentOptimal && newLayout == vk::ImageLayout::ePresentSrcKHR) {
        srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput;
        dstStageMask = vk::PipelineStageFlagBits2::eBottomOfPipe;
        srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite;
        dstAccessMask = {};
    } else {
        throw std::invalid_argument("unsupported layout transition!");
    }

    transitionImageLayout(commandBuffer, image, srcStageMask, dstStageMask, srcAccessMask, dstAccessMask, oldLayout, newLayout, subresourceRange);
}

void transitionImageLayout(const vk::raii::CommandBuffer& commandBuffer,
                           vk::Image image,
                           vk::PipelineStageFlagBits2 srcStageMask,
                           vk::PipelineStageFlagBits2 dstStageMask,
                           vk::Flags<vk::AccessFlagBits2> srcAccessMask,
                           vk::Flags<vk::AccessFlagBits2> dstAccessMask,
                           vk::ImageLayout oldLayout,
                           vk::ImageLayout newLayout,
                           const vk::ImageSubresourceRange& subresourceRange)
{
    vk::ImageMemoryBarrier2 barrier {
        .srcStageMask = srcStageMask,
        .srcAccessMask = srcAccessMask,
        .dstStageMask = dstStageMask,
        .dstAccessMask = dstAccessMask,
        .oldLayout = oldLayout,
        .newLayout = newLayout,
        .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
        .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
        .image = image,
        .subresourceRange = subresourceRange,
    };
    vk::DependencyInfo dependencyInfo {
        .dependencyFlags = {},
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &barrier,
    };
    commandBuffer.pipelineBarrier2(dependencyInfo);
}