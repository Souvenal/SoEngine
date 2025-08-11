#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <cstdint>
#include <optional>
#include <vector>


struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily; // support drawing commands
    std::optional<uint32_t> presentFamily;  // support presentation
    std::optional<uint32_t> transferFamily; // support transfer operations

    bool isComplete() {
        return
            graphicsFamily.has_value() &&
            presentFamily.has_value() &&
            transferFamily.has_value();
    }
};

struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};