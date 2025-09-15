#pragma once

#include "platform/input_events.h"
#include "platform/window.h"
#include "utils/deletion_queue.h"
#include "render/vulkan/vk_types.h"
#include "render/vulkan/vk_descriptors.h"
#include "utils/logging.h"
#include "utils/type_traits.h"
#include "common/error.h"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

#include <set>
#include <memory>
#include <filesystem>

struct AppInfo {
    const uint32_t maxFramesInFlight { 2 };
    vk::SampleCountFlagBits msaaSamples { vk::SampleCountFlagBits::e1 };
    bool dynamicRenderingSupported      { false };
    bool timelineSemaphoresSupported    { false };
    bool synchronization2Supported      { false };

    std::vector<const char*> requiredDeviceExtensions = {
    #ifdef __APPLE__
        "VK_KHR_portability_subset",
    #endif
        vk::KHRSwapchainExtensionName,
        // vk::KHRSpirv14ExtensionName,
        // vk::KHRCreateRenderpass2ExtensionName,
        // vk::KHRCopyCommands2ExtensionName,
        // vk::KHRBufferDeviceAddressExtensionName
    };

    virtual void checkFeatureSupport(vk::Instance instance, vk::PhysicalDevice physicalDevice);

    virtual void detectFeatureSupport(vk::PhysicalDevice physicalDevice);
};

struct AppCapabilitiesSummary {
    std::string gpuName;
    uint32_t apiVersionMajor{};
    uint32_t apiVersionMinor{};
    uint32_t apiVersionPatch{};
    uint32_t swapImageCount{};
    vk::PresentModeKHR presentMode{};
    vk::Format swapFormat{};
    bool dynamicRendering{};
    bool timelineSemaphores{};
    bool sync2{};
    bool profileSupported{};
};

class Application {
public:

    Application() = delete;
    explicit Application(
            const std::filesystem::path& appDir,
            const Window* window,
            const AppInfo& appInfo = AppInfo{});
    virtual ~Application() = 0;
    virtual void onInit();
    virtual void onUpdate(double deltaTime);
    virtual void onRender();
    virtual void onInputEvent(const InputEvent& event);
    virtual void onShutdown();

    [[nodiscard]] virtual bool shouldTerminate() const;
    [[nodiscard]] virtual bool isSwapchainOutOfDate() const;

    /**
     * @brief Recreates the swap chain
     * @note Should be called when the swap chain becomes outdated (e.g., due to window resizing)
     */
    virtual void recreateSwapchain();

protected:

    std::filesystem::path appDir;
    AppInfo appInfo;

    const Window* window  { nullptr };

    AppCapabilitiesSummary capsSummary {};

    DeletionQueue       mainDeletionQueue;
    DeletionQueue       resourceDeletionQueue;

    vk::raii::Context                           context {};
    std::unique_ptr<vk::raii::Instance>         instance;
    vk::raii::DebugUtilsMessengerEXT            debugMessenger { nullptr };

    vk::raii::SurfaceKHR                        surface { nullptr };

    std::unique_ptr<vk::raii::PhysicalDevice>   physicalDevice {};
    std::unique_ptr<vk::raii::Device>           device { nullptr };

    QueueFamilyIndices  queueFamilyIndices {};
    vk::raii::Queue     graphicsQueue   { nullptr };
    vk::raii::Queue     presentQueue    { nullptr };
    vk::raii::Queue     computeQueue    { nullptr };
    vk::raii::Queue     transferQueue   { nullptr };

    MemoryAllocator         memoryAllocator {};
    DescriptorAllocator     globalDescriptorAllocator {};

    uint32_t    minImageCountInSwapchain {0};
    vk::raii::SwapchainKHR              swapchain { nullptr };
    std::vector<vk::Image>              swapchainImages;
    std::vector<vk::raii::ImageView>    swapchainImageViews;
    vk::Format                          swapchainImageFormat {};
    vk::Extent2D                        swapchainExtent;
    bool                                swapchainOutOfDate {false};

    // Traditional render pass (fallback for non-dynamic rendering)
    vk::raii::RenderPass                renderPass { nullptr };
    std::vector<vk::raii::Framebuffer>  swapChainFramebuffers;

    vk::raii::CommandPool                   graphicsCommandPool { nullptr };
    vk::raii::CommandPool                   computeCommandPool  { nullptr };
    vk::raii::CommandPool                   transferCommandPool { nullptr };

    AllocatedImage                 drawImage;
    vk::raii::DescriptorSetLayout  drawImageDescriptorSetLayout { nullptr };
    vk::raii::DescriptorSets       drawImageDescriptorSets { nullptr };

    vk::raii::Semaphore                 semaphore   { nullptr };
    uint64_t                            timelineValue {0};
    std::vector<vk::raii::Fence>        inFlightFences;

    size_t                  currentFrame {0};

protected:

    /**
     * @brief Initializes the Vulkan instance
     * @param appName Name of the application
     */
    virtual void initInstance(const std::string& appName);

    /**
     * @brief Initializes the debug messenger for Vulkan validation layers
     * @note This function should be called after the Vulkan instance is created
     *       and before any Vulkan objects are created
     */
    virtual void initDebugMessenger();

    /**
     * @brief Initializes the Vulkan surface for rendering
     */
    virtual void initSurface();
    
    /**
     * @brief Selects a physical device (GPU) that supports the required features and extensions
     * @note This function should be called after the Vulkan instance and surface are created
     */
    virtual void selectPhysicalDevice();

    /**
     * @brief Initializes the logical device and retrieves the graphics and present queues
     * @note A default implementation is provided that initializes the logical device
     *       without any additional features. Derived classes can override this method
     */
    virtual void initLogicalDevice();
    
    /**
     * @brief Initializes the logical device and retrieves the graphics and present queues
     * @tparam FeatureChainT Type of the feature chain (e.g., vk::StructureChain<...>)
     * @param featureChain The feature chain containing the desired device features
     */
    template<typename... T>
    void initLogicalDevice(const vk::StructureChain<T...>& featureChain);

    /**
     * @brief Initializes the memory allocator for managing Vulkan memory
     * @note This function should be called after the logical device is created
     */
    virtual void initMemoryAllocator();

    /**
     * @brief Initializes the swap chain images and image views
     */
    virtual void initSwapchain();

    /**
     * @brief Initializes the draw image used for rendering
     */
    virtual void initDrawImage();

    /**
     * @brief Initializes the command pools for graphics, compute, and transfer operations
     * @note This function should be called after the logical device is created
     */
    virtual void initCommandPools();

    /**
     * @brief Initializes the command buffers for graphics, compute, and transfer operations
     * @note This function should be called after the command pools are created
     */
    virtual void initCommandBuffers();

    /**
     * @brief Initializes the render pass
     * @note If dynamic rendering is supported, this function can be a no-op
     */
    virtual void initRenderPass();

    /**
     * @brief Initializes the framebuffers
     * @note If dynamic rendering is supported, this function can be a no-op
     */
    virtual void initFramebuffers();

    /**
     * @brief Initializes the descriptor allocator for managing Vulkan descriptor sets
     */
    virtual void initDescriptorAllocator();

    /**
     * @brief Initializes the descriptor set layouts
     */
    virtual void initDescriptorSetLayouts();

    /**
     * @brief Initializes the descriptor set layout for the draw image
     */
    virtual void initDrawImageDescriptorSetLayout();

    /**
     * @brief Initializes the graphics pipeline
     * @note This function should set up the pipeline layouts and pipelines
     */
    virtual void initPipelines();

    /**
     * @brief Initializes the descriptor sets
     */
    virtual void initDescriptorSets();

    /**
     * @brief Initializes the descriptor sets for the draw image
     */
    virtual void initDrawImageDescriptorSets();

    /**
     * @brief Initializes synchronization objects (semaphores and fences)
     */
    virtual void initSyncObjects();

    /**
     * @brief Initializes ImGui for GUI rendering
     */
    virtual void initImGui();

    /**
     * @brief Builds and a summary of the application's capabilities
     */
    virtual void buildCapabilitiesSummary();

    /**
     * @brief Logs the application's capabilities summary
     */
    virtual void logCapabilitiesSummary() const;

    /**
     * @brief Updates the ImGui interface
     */
    virtual void updateImGui();

    /**
     * @brief Draws a single frame
     */
    virtual void drawFrame() = 0;

    /**
     * @brief Draws the background
     */
    virtual void drawBackground(const vk::raii::CommandBuffer& commandBuffer);

    /**
     * @brief Draws the 3D scene
     */
    virtual void drawScene(const vk::raii::CommandBuffer& commandBuffer, uint32_t imageIndex);

    /**
     * @brief Draws the ImGui interface
     */
    virtual void drawImGui(const vk::raii::CommandBuffer& commandBuffer, uint32_t imageIndex);

    /**
     * @brief Cleans up Vulkan resources
     */
    virtual void cleanup();

    /**
     * @brief Cleans up swap chain related resources
     */
    virtual void cleanupSwapchain();

    /**
     * Helper functions
     */
    [[nodiscard]] std::vector<const char*> getRequiredLayers() const;
    void checkLayerSupport(const std::vector<const char*>& requiredLayers) const;
    [[nodiscard]] std::vector<const char*> getRequiredExtensions() const;
    void checkExtensionSupport(const std::vector<const char*>& requiredExtensions) const;

    /**
     * @brief Dumps the contents of an AllocatedBuffer to a binary file
     */
    void dumpAllocatedBuffer(
        const AllocatedBuffer& buffer,
        vk::DeviceSize bufferSize,
        const std::string& filename);
};

template<typename... T>
void Application::initLogicalDevice(const vk::StructureChain<T...>& featureChain) {
    ASSERT(physicalDevice, "Physical device must be selected before creating logical device");

    const auto queueFamilyProperties = physicalDevice->getQueueFamilyProperties();
    queueFamilyIndices = vkutil::findQueueFamilies(*physicalDevice, surface);
    uint32_t graphicsIndex = queueFamilyIndices.graphicsFamily.value();
    uint32_t presentIndex = queueFamilyIndices.presentFamily.value();
    uint32_t computeIndex = queueFamilyIndices.computeFamily.value();
    uint32_t transferIndex = queueFamilyIndices.transferFamily.value();

    std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
    std::set<uint32_t> uniqueQueueFamilies = {
        graphicsIndex, presentIndex, computeIndex, transferIndex
    }; // if queue families are the same, only need to pass its index once

    float queuePriority = 0.0f;
    for (uint32_t queueFamily : uniqueQueueFamilies) {
        vk::DeviceQueueCreateInfo queueCreateInfo {
            .queueFamilyIndex = queueFamily,
            .queueCount = 1,
            .pQueuePriorities = &queuePriority
        };
        queueCreateInfos.emplace_back(queueCreateInfo);
    }

    using HeadT = first_type_t<T...>;
    vk::DeviceCreateInfo deviceCreateInfo {
        .pNext = &featureChain.template get<HeadT>(),
        .queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size()),
        .pQueueCreateInfos = queueCreateInfos.data(),
        .enabledExtensionCount = static_cast<uint32_t>(appInfo.requiredDeviceExtensions.size()),
        .ppEnabledExtensionNames = appInfo.requiredDeviceExtensions.data(),
    };
    device = std::make_unique<vk::raii::Device>( *physicalDevice, deviceCreateInfo );

    LOG_CORE_DEBUG("Logical device is successfully initialized");

    graphicsQueue = vk::raii::Queue(*device, graphicsIndex, 0);
    presentQueue = vk::raii::Queue(*device, presentIndex, 0);
    computeQueue = vk::raii::Queue(*device, computeIndex, 0);
    transferQueue = vk::raii::Queue(*device, transferIndex, 0);

    LOG_CORE_DEBUG("Queues are successfully initialized with queue family index: Graphics({}), Present({}), Compute({}), Transfer({})",
                   graphicsIndex, presentIndex, computeIndex, transferIndex);
}