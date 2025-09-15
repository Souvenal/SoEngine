#include "vk_images.h"

#include "utils/logging.h"
#include "vk_initializers.h"


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

    // Undefined -> General (for storage image or general usage)
    if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eGeneral) {
        srcStageMask = vk::PipelineStageFlagBits2::eTopOfPipe;
        dstStageMask = vk::PipelineStageFlagBits2::eAllCommands;
        srcAccessMask = {};
        dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite;
    }
    // Undefined -> TransferDstOptimal (for copying from staging buffer)
    else if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eTransferDstOptimal) {
        srcStageMask = vk::PipelineStageFlagBits2::eTopOfPipe;
        dstStageMask = vk::PipelineStageFlagBits2::eTransfer;
        srcAccessMask = {};
        dstAccessMask = vk::AccessFlagBits2::eTransferWrite;
    }
    //  Undefined -> DepthStencilAttachmentOptimal (for depth buffer)
    else if (oldLayout == vk::ImageLayout::eUndefined &&
               (newLayout == vk::ImageLayout::eDepthStencilAttachmentOptimal ||
                newLayout == vk::ImageLayout::eDepthAttachmentOptimal)) {
        srcStageMask = vk::PipelineStageFlagBits2::eTopOfPipe;
        dstStageMask = vk::PipelineStageFlagBits2::eEarlyFragmentTests;
        srcAccessMask = {};
        dstAccessMask = vk::AccessFlagBits2::eDepthStencilAttachmentRead | vk::AccessFlagBits2::eDepthStencilAttachmentWrite;
    }
    // General -> TransferSrcOptimal (after written by compute shader, before being read by transfer)
    else if (oldLayout == vk::ImageLayout::eGeneral && newLayout == vk::ImageLayout::eTransferSrcOptimal) {
        srcStageMask = vk::PipelineStageFlagBits2::eComputeShader;
        dstStageMask = vk::PipelineStageFlagBits2::eTransfer;
        srcAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite;
        dstAccessMask = vk::AccessFlagBits2::eTransferRead;
    }
    // TransferDstOptimal -> ShaderReadOnlyOptimal (for sampling from the image)
    else if (oldLayout == vk::ImageLayout::eTransferDstOptimal && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal) {
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

void copyBufferToImage(
        const vk::raii::CommandBuffer& commandBuffer,
        const vk::raii::Buffer& buffer,
        const vk::raii::Image& image,
        uint32_t width,
        uint32_t height) {
    ASSERT(*buffer, "Invalid buffer");
    ASSERT(*image, "Invalid image");
    vk::BufferImageCopy region {
        .bufferOffset = 0,
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource = {
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .mipLevel = 0,
            .baseArrayLayer = 0,
            .layerCount = 1},
        .imageOffset = {0, 0, 0},
        .imageExtent = {
            .width = width,
            .height = height,
            .depth = 1}};
    commandBuffer.copyBufferToImage(buffer, image, vk::ImageLayout::eTransferDstOptimal, {region});
}

void generateMipmaps(
        const vk::raii::PhysicalDevice& physicalDevice,
        const vk::raii::CommandBuffer& commandBuffer,
        const vk::raii::Image& image,
        vk::Format imageFormat,
        int32_t texWidth,
        int32_t texHeight,
        uint32_t mipLevels) {
    ASSERT(*physicalDevice, "Invalid physical device");
    ASSERT(*commandBuffer, "Invalid command buffer");
    ASSERT(*image, "Invalid image");
    // Check if the image format supports linear blitting
    vk::FormatProperties formatProperties = physicalDevice.getFormatProperties(imageFormat);
    if (!(formatProperties.optimalTilingFeatures & vk::FormatFeatureFlagBits::eSampledImageFilterLinear)) {
        throw std::runtime_error("Texture image format does not support linear blitting!");
    }
    // alternatives:
    // 1.   search common texture image formats for one
    //      that does support linear bitting
    // 2.   implement the mipmap generation in software with a library like stb_image_resize

    vk::ImageMemoryBarrier barrier {
        .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
        .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
        .image = image,
        .subresourceRange = {
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        }
    };

    int32_t mipWidth {texWidth}, mipHeight {texHeight};
    for (uint32_t i = 1; i < mipLevels; ++i) {
        barrier.subresourceRange.baseMipLevel = i - 1;
        barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
        barrier.newLayout = vk::ImageLayout::eTransferSrcOptimal;
        barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
        barrier.dstAccessMask = vk::AccessFlagBits::eTransferRead;

        commandBuffer.pipelineBarrier(
            vk::PipelineStageFlagBits::eTransfer,
            vk::PipelineStageFlagBits::eTransfer,
            {}, {}, {}, barrier);

        vk::ArrayWrapper1D<vk::Offset3D, 2> srcOffsets, dstOffsets;
        srcOffsets[0] = {0, 0, 0};
        srcOffsets[1] = {mipWidth, mipHeight, 1};
        dstOffsets[0] = {0, 0, 0};
        dstOffsets[1] = {mipWidth > 1 ? mipWidth / 2 : 1, mipHeight > 1 ? mipHeight / 2 : 1, 1};
        vk::ImageBlit blit {
            .srcSubresource = {vk::ImageAspectFlagBits::eColor, i -1 , 0, 1},
            .srcOffsets = srcOffsets,
            .dstSubresource = {vk::ImageAspectFlagBits::eColor, i, 0, 1},
            .dstOffsets = dstOffsets,
        };
        commandBuffer.blitImage(
            image, vk::ImageLayout::eTransferSrcOptimal,
            image, vk::ImageLayout::eTransferDstOptimal,
            {blit}, vk::Filter::eLinear);

        barrier.oldLayout = vk::ImageLayout::eTransferSrcOptimal;
        barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        barrier.srcAccessMask = vk::AccessFlagBits::eTransferRead;
        barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

        commandBuffer.pipelineBarrier(
            vk::PipelineStageFlagBits::eTransfer,
            vk::PipelineStageFlagBits::eFragmentShader,
            {}, {}, {}, barrier);

        if (mipWidth > 1) mipWidth /= 2;
        if (mipHeight > 1) mipHeight /= 2;
    }

    barrier.subresourceRange.baseMipLevel = mipLevels - 1;
    barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
    barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
    barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

    commandBuffer.pipelineBarrier(
        vk::PipelineStageFlagBits::eTransfer,
        vk::PipelineStageFlagBits::eFragmentShader,
        {}, {}, {}, barrier);
}

}   // namespace vkutil