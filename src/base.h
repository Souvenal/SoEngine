#pragma once

#ifdef __INTELLISENSE__
#include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif

struct AppInfo {
    bool dynamicRenderingSupported      { false };
    bool timelineSemaphoresSupported    { false };
    bool synchronization2Supported      { false };

    void printFeatureSupportSummary() {
        // Print feature support summary
        std::cout << "\nFeature support summary:\n";
        std::cout << "- Dynamic Rendering: " << (dynamicRenderingSupported ? "Yes" : "No") << "\n";
        std::cout << "- Timeline Semaphores: " << (timelineSemaphoresSupported ? "Yes" : "No") << "\n";
        std::cout << "- Synchronization2: " << (synchronization2Supported ? "Yes" : "No") << "\n";
    }
};