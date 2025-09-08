#pragma once

#include "common/vk_common.h"

#include <vector>

namespace vkutil
{

/**
 * @brief Checks if a physical device is suitable for the application's needs
 */
[[nodiscard]] bool isDeviceSuitable(
    const vk::raii::PhysicalDevice& device,
    const vk::raii::SurfaceKHR& surface,
    const std::vector<const char*>& requiredDeviceExtensions);

/** 
 * @brief Rates the suitability of a physical device for the application's needs
 * @note Calls isDeviceSuitable internally
 * @return A score representing the suitability of the device (0 if not suitable)
 */
[[nodiscard]] uint32_t rateDeviceSuitability(
        const vk::raii::PhysicalDevice& physicalDevice,
        const vk::raii::SurfaceKHR& surface,
        const std::vector<const char*>& requiredDeviceExtensions);

/**
 * @brief Gets the maximum usable sample count for MSAA
 */
[[nodiscard]] vk::SampleCountFlagBits getMaxUsableSampleCount(
        const vk::PhysicalDevice& physicalDevice) noexcept;

/**
 * @brief Chooses the best surface format from the available options
 */
[[nodiscard]] vk::Format chooseSurfaceFormat(
        const std::vector<vk::SurfaceFormatKHR>& availableFormats) noexcept;

/**
 * @brief Chooses the best present mode from the available options
 */
[[nodiscard]] vk::PresentModeKHR chooseSwapPresentMode(
        const std::vector<vk::PresentModeKHR>& availablePresentModes) noexcept;

/**
 * @brief Finds a supported format from the given candidates
 */
[[nodiscard]] vk::Format findSupportedFormat(
        const vk::raii::PhysicalDevice& physicalDevice,
        const std::vector<vk::Format>& candidates,
        vk::ImageTiling tiling,
        vk::FormatFeatureFlags features);

/**
 * @brief Finds the depth format for the given physical device
 */
[[nodiscard]] vk::Format findDepthFormat(
        const vk::raii::PhysicalDevice& physicalDevice);

/**
 * @brief Finds a memory type index on the physical device that satisfies the requested properties
 */
[[nodiscard]] uint32_t findMemoryType(
        const vk::raii::PhysicalDevice& physicalDevice,
        uint32_t typeFilter,
        vk::MemoryPropertyFlags properties);

/**
 * @brief Begins a single time command buffer
 */
[[nodiscard]] vk::raii::CommandBuffer beginSingleTimeCommands(
        const vk::raii::Device& device,
        const vk::raii::CommandPool& commandPool);

/**
 * @brief Ends the single time command buffer and submits it to the given queue
 */
void endSingleTimeCommands(
        const vk::raii::CommandBuffer& commandBuffer,
        const vk::raii::Queue& queue);

} // namespace vkutil