#pragma once

#ifdef __INTELLISENSE__
#include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif

#include <print>

struct AppInfo {
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