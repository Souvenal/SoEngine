#include "vk_utils.h"

#include "utils/logging.h"

#include <ranges>
#include <algorithm>

namespace vkutil
{

bool isDeviceSuitable(
        const vk::raii::PhysicalDevice& physicalDevice,
        const vk::raii::SurfaceKHR& surface,
        const std::vector<const char*>& requiredDeviceExtensions) {
    ASSERT(*physicalDevice, "Invalid physical device");
    ASSERT(*surface, "Invalid surface");
    auto availableExtensions = physicalDevice.enumerateDeviceExtensionProperties();
    auto deviceProperties = physicalDevice.getProperties();
    auto deviceFeatures = physicalDevice.getFeatures();
    auto queueFamilies = physicalDevice.getQueueFamilyProperties();

    bool extensionSupported = std::ranges::all_of(requiredDeviceExtensions,
        [&availableExtensions](const char* deviceExtension) {
            return std::ranges::any_of(availableExtensions,
                [&deviceExtension](const auto& availableExtension) {
                    return strcmp(deviceExtension,
                        availableExtension.extensionName) == 0;
                });
        });
    bool queueFamilySupported = std::ranges::any_of(queueFamilies,
        [](const vk::QueueFamilyProperties& qfp) {
            return (qfp.queueFlags & vk::QueueFlagBits::eGraphics) != static_cast<vk::QueueFlagBits>(0);
        });
    bool swapChainAdequate = false;
    if (extensionSupported) {
        auto formats = physicalDevice.getSurfaceFormatsKHR(surface);
        auto presentModes = physicalDevice.getSurfacePresentModesKHR(surface);
        // at least one format and one present mode must be available
        swapChainAdequate = !formats.empty() && !presentModes.empty();
    }

    return deviceProperties.apiVersion >= vk::ApiVersion13 &&
        queueFamilySupported &&
        extensionSupported &&
        swapChainAdequate &&
        deviceFeatures.samplerAnisotropy;
}

uint32_t rateDeviceSuitability(
        const vk::raii::PhysicalDevice& physicalDevice,
        const vk::raii::SurfaceKHR& surface,
        const std::vector<const char*>& requiredDeviceExtensions) {
    ASSERT(*physicalDevice, "Invalid physical device");
    if (!isDeviceSuitable(physicalDevice, surface, requiredDeviceExtensions)) {
        return 0;
    }

    const auto deviceProperties = physicalDevice.getProperties();

    uint32_t score = 0;
    // discrete GPUs have significant performance advantage
    if (deviceProperties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu) {
        score += 1000;
    }

    score += deviceProperties.limits.maxImageDimension2D;

    return score;
}

vk::SampleCountFlagBits getMaxUsableSampleCount(
        const vk::PhysicalDevice& physicalDevice) noexcept {
    ASSERT(physicalDevice, "Invalid physical device");
    const auto physicalDeviceProperties = physicalDevice.getProperties();

    vk::SampleCountFlags counts {
        physicalDeviceProperties.limits.framebufferColorSampleCounts &
        physicalDeviceProperties.limits.framebufferDepthSampleCounts
    };

    if (counts & vk::SampleCountFlagBits::e64) {
        return vk::SampleCountFlagBits::e64;
    } else if (counts & vk::SampleCountFlagBits::e32) {
        return vk::SampleCountFlagBits::e32;
    } else if (counts & vk::SampleCountFlagBits::e16) {
        return vk::SampleCountFlagBits::e16;
    } else if (counts & vk::SampleCountFlagBits::e8) {
        return vk::SampleCountFlagBits::e8;
    } else if (counts & vk::SampleCountFlagBits::e4) {
        return vk::SampleCountFlagBits::e4;
    } else if (counts & vk::SampleCountFlagBits::e2) {
        return vk::SampleCountFlagBits::e2;
    } else {
        return vk::SampleCountFlagBits::e1;
    }
}

vk::Format chooseSurfaceFormat(
        const std::vector<vk::SurfaceFormatKHR>& availableFormats) noexcept {
    ASSERT(!availableFormats.empty(), "No available surface formats");
    for (const auto& availableFormat : availableFormats) {
        if (availableFormat.format == vk::Format::eB8G8R8A8Srgb &&
            availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear
        ) {
            return availableFormat.format;
        }
    }

    return availableFormats[0].format == vk::Format::eUndefined ?
        vk::Format::eB8G8R8A8Unorm : availableFormats[0].format;
}

vk::PresentModeKHR chooseSwapPresentMode(
        const std::vector<vk::PresentModeKHR>& availablePresentModes) noexcept {
    for (const auto& availablePresentMode : availablePresentModes) {
        if (availablePresentMode == vk::PresentModeKHR::eMailbox) {
            // relatively high energy consumption, but low latency
            return availablePresentMode; // prefer mailbox mode for lower latency
        }
    }

    return vk::PresentModeKHR::eFifo;
}

vk::Format findSupportedFormat(
        const vk::raii::PhysicalDevice& physicalDevice,
        const std::vector<vk::Format>& candidates,
        vk::ImageTiling tiling,
        vk::FormatFeatureFlags features) {
    ASSERT(*physicalDevice, "Invalid physical device");
    for (const auto format : candidates) {
        vk::FormatProperties props = physicalDevice.getFormatProperties(format);
        if (tiling == vk::ImageTiling::eLinear &&
            (props.linearTilingFeatures & features) == features) {
            return format;
        } else if (tiling == vk::ImageTiling::eOptimal &&
            props.optimalTilingFeatures & features) {
            return format;
        }
    }

    throw std::runtime_error("Failed to find supported format!");
}

vk::Format findDepthFormat(const vk::raii::PhysicalDevice& physicalDevice) {
    ASSERT(*physicalDevice, "Invalid physical device");
    return findSupportedFormat(
        physicalDevice,
        {vk::Format::eD32Sfloat,
         vk::Format::eD32SfloatS8Uint,
         vk::Format::eD24UnormS8Uint},
        vk::ImageTiling::eOptimal,
        vk::FormatFeatureFlagBits::eDepthStencilAttachment);
}

uint32_t findMemoryType(
        const vk::raii::PhysicalDevice& physicalDevice,
        uint32_t typeFilter,
        vk::MemoryPropertyFlags properties) {
    ASSERT(*physicalDevice, "Invalid physical device");
    auto memProperties = physicalDevice.getMemoryProperties();

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; ++i) {
        if (typeFilter & (1 << i) &&
            (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    throw std::runtime_error("Failed to find suitable memory type!");
}

vk::raii::CommandBuffer beginSingleTimeCommands(
        const vk::raii::Device& device,
        const vk::raii::CommandPool& commandPool) {
    ASSERT(*device, "Invalid device");
    ASSERT(*commandPool, "Invalid command pool");
    vk::CommandBufferAllocateInfo allocInfo{
        .commandPool = commandPool,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = 1};

    vk::raii::CommandBuffer commandBuffer{
        std::move(device.allocateCommandBuffers(allocInfo).front())};
    commandBuffer.begin(
        vk::CommandBufferBeginInfo{
            .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit
        });

    return commandBuffer;
}

void endSingleTimeCommands(
        const vk::raii::CommandBuffer& commandBuffer,
        const vk::raii::Queue& queue) {
    ASSERT(*commandBuffer, "Invalid command buffer");
    ASSERT(*queue, "Invalid queue");
    commandBuffer.end();

    queue.submit(
        vk::SubmitInfo{
            .commandBufferCount = 1,
            .pCommandBuffers = &*commandBuffer},
        nullptr);
    queue.waitIdle();
}

}   // namespace vkutil