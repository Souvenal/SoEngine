#pragma once

#include <vulkan/vulkan_profiles.hpp>
#include <print>

constexpr uint32_t WIDTH {800};
constexpr uint32_t HEIGHT {600};
constexpr size_t MAX_FRAMES_IN_FLIGHT { 3 };
constexpr size_t MAX_OBJECTS { 3 };

struct AppInfo {
    bool profileSupported {false};
    VpProfileProperties profile;
    bool dynamicRenderingSupported      { false };
    bool timelineSemaphoresSupported    { false };
    bool synchronization2Supported      { false };

    void printFeatureSupportSummary() {
        // Print feature support summary
        std::println("Feature support summary:");
        std::println("- Dynamic Rendering:      {}", (dynamicRenderingSupported ? "Yes" : "No"));
        std::println("- Timeline Semaphores:    {}", (timelineSemaphoresSupported ? "Yes" : "No"));
        std::println("- Synchronization2:       {}", (synchronization2Supported ? "Yes" : "No"));
    }
};