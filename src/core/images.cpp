#include "images.h"

#include <stdexcept>

#include "initializers.h"

namespace vkutil
{

vk::raii::ImageView createImageView(const vk::raii::Device& device,
                                        vk::Image image,
                                        vk::Format format,
                                        vk::ImageAspectFlags aspectFlags,
                                        uint32_t mipLevels)
{
    auto viewInfo = vkinit::imageViewCreateInfo(image, format, aspectFlags, mipLevels);
    return device.createImageView(viewInfo);
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
    } else if (oldLayout == vk::ImageLayout::eUndefined &&
               (newLayout == vk::ImageLayout::eDepthStencilAttachmentOptimal ||
                newLayout == vk::ImageLayout::eDepthAttachmentOptimal)) {
        srcStageMask = vk::PipelineStageFlagBits2::eTopOfPipe;
        dstStageMask = vk::PipelineStageFlagBits2::eEarlyFragmentTests;
        srcAccessMask = {};
        dstAccessMask = vk::AccessFlagBits2::eDepthStencilAttachmentRead | vk::AccessFlagBits2::eDepthStencilAttachmentWrite;
    } else if (oldLayout == vk::ImageLayout::eTransferDstOptimal && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal) {
        srcStageMask = vk::PipelineStageFlagBits2::eTransfer;
        dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader;
        srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
        dstAccessMask = vk::AccessFlagBits2::eShaderRead;
    } else if (oldLayout == vk::ImageLayout::eTransferDstOptimal && newLayout == vk::ImageLayout::eColorAttachmentOptimal) {
        srcStageMask = vk::PipelineStageFlagBits2::eTransfer;
        dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput;
        srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
        dstAccessMask = vk::AccessFlagBits2::eColorAttachmentRead | vk::AccessFlagBits2::eColorAttachmentWrite;
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
    } else if (oldLayout == vk::ImageLayout::eColorAttachmentOptimal && newLayout == vk::ImageLayout::eTransferSrcOptimal) {
        srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput;
        dstStageMask = vk::PipelineStageFlagBits2::eTransfer;
        srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite;
        dstAccessMask = vk::AccessFlagBits2::eTransferRead;
    } else if (oldLayout == vk::ImageLayout::eTransferDstOptimal && newLayout == vk::ImageLayout::ePresentSrcKHR) {
        srcStageMask = vk::PipelineStageFlagBits2::eTransfer;
        dstStageMask = vk::PipelineStageFlagBits2::eBottomOfPipe;
        srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
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

void copyImageToImage(const vk::raii::CommandBuffer& commandBuffer,
                      vk::Image srcImage,
                      vk::Image dstImage,
                      const vk::Extent2D& srcSize,
                      const vk::Extent2D& dstSize)
{
    // Blit image lets you copy images of different formats and different sizes into one another
    vk::ImageBlit2 blitRegion {};

    blitRegion.srcOffsets[1].x = srcSize.width;
    blitRegion.srcOffsets[1].y = srcSize.height;
    blitRegion.srcOffsets[1].z = 1;

    blitRegion.dstOffsets[1].x = dstSize.width;
    blitRegion.dstOffsets[1].y = dstSize.height;
    blitRegion.dstOffsets[1].z = 1;

    blitRegion.srcSubresource = { vk::ImageAspectFlagBits::eColor, 0, 0, 1 };
    blitRegion.dstSubresource = { vk::ImageAspectFlagBits::eColor, 0, 0, 1 };

    vk::BlitImageInfo2 blitInfo {
        .srcImage = srcImage,
        .srcImageLayout = vk::ImageLayout::eTransferSrcOptimal,
        .dstImage = dstImage,
        .dstImageLayout = vk::ImageLayout::eTransferDstOptimal,
        .regionCount = 1,
        .pRegions = &blitRegion,
        .filter = vk::Filter::eLinear
    };
    commandBuffer.blitImage2(blitInfo);
}

}   // namespace vkutil