#pragma once

#include "common/vk_common.h"

struct AllocatedImage {
    vk::Image image;
    vk::ImageView imageView   { nullptr };
    VmaAllocation allocation;
    vk::Extent3D imageExtent;
    vk::Format imageFormat;
};

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
vk::raii::ImageView createImageView(const vk::raii::Device& device,
                                    vk::Image image,
                                    vk::Format format,
                                    vk::ImageAspectFlags aspectFlags,
                                    uint32_t mipLevels);

/**
 * @brief Uses VMA to create an allocated image.
 * @note Should use vmaDestroyImage to free the image.
 */
[[nodiscard]]
AllocatedImage createAllocatedImage(const vk::raii::Device& device,
                                    VmaAllocator allocator,
                                    vk::Extent3D extent,
                                    uint32_t mipLevels,
                                    vk::SampleCountFlagBits numSamples,
                                    vk::Format format,
                                    vk::ImageTiling tiling,
                                    vk::ImageUsageFlags usage,
                                    vk::MemoryPropertyFlags properties);


/**
 * @brief Transition the layout of an image, on a fixed subresource with first miplevel and layer.
 *
 * The srcStageMask, dstStageMask, srcAccessMask and dstAccessMask are determined from the old and new layouts.
 */
void transitionImageLayout(const vk::raii::CommandBuffer& commandBuffer,
                           vk::Image image,
                           vk::ImageLayout oldLayout,
                           vk::ImageLayout newLayout);

/**
 * @brief Transition the layout of an image, on a given subresource with first miplevel and layer.
 *
 * The srcStageMask, dstStageMask, srcAccessMask and dstAccessMask are determined from the old and new layouts.
 */
void transitionImageLayout(const vk::raii::CommandBuffer& commandBuffer,
                           vk::Image image,
                           vk::ImageLayout oldLayout,
                           vk::ImageLayout newLayout,
                           const vk::ImageSubresourceRange& subresourceRange);

/**
 * @brief Transition the layout of an image, on a given subresource with first miplevel and layer.
 *
 * The srcStageMask, dstStageMask, srcAccessMask and dstAccessMask need to be specified.
 */
void transitionImageLayout(const vk::raii::CommandBuffer& commandBuffer,
                           vk::Image image,
                           vk::PipelineStageFlagBits2 srcStageMask,
                           vk::PipelineStageFlagBits2 dstStageMask,
                           vk::Flags<vk::AccessFlagBits2> srcAccessMask,
                           vk::Flags<vk::AccessFlagBits2> dstAccessMask,
                           vk::ImageLayout oldLayout,
                           vk::ImageLayout newLayout,
                           const vk::ImageSubresourceRange& subresourceRange);