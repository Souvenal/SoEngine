#pragma once

#include "types.h"

namespace vkutil
{

[[nodiscard]]
vk::raii::ImageView createImageView(const vk::raii::Device& device,
                                    vk::Image image,
                                    vk::Format format,
                                    vk::ImageAspectFlags aspectFlags,
                                    uint32_t mipLevels);


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

/**
 * @brief Copies an image to another image.
 */
void copyImageToImage(const vk::raii::CommandBuffer& commandBuffer,
                      vk::Image srcImage,
                      vk::Image dstImage,
                      const vk::Extent2D& srcSize,
                      const vk::Extent2D& dstSize);

}   // namespace vkutil