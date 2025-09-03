#pragma once

#include "utils/deletion_queue.h"
#include "window/window.h"
#include "window/input_events.h"
#include "resource/model_object.h"
#include "config.h"
#include "pipelines.h"
#include "descriptors.h"
#include "images.h"
#include "frame_data.h"

#include <ktxvulkan.h>

#include <limits>       // std::numeric_limits
#include <vector>
#include <memory>
#include <filesystem>


class SoEngine {
public:
    explicit SoEngine(const std::filesystem::path& path);

    void prepare(const Window* window);

    void update(float deltaTime);

    [[nodiscard]] bool shouldTerminate() const;

    void terminate();

    [[nodiscard]] bool isSwapChainOutOfDate() const;

    /**
     * @brief Recreates the swap chain
     * @note Should be called when the swap chain becomes outdated (e.g., due to window resizing)
     */
    void recreateSwapChain();

    void inputEvent(const InputEvent& event);

    FrameData& getCurrentFrame();

private:
    AppInfo appInfo;
    std::filesystem::path appDir;

    DeletionQueue       mainDeletionQueue;
    DeletionQueue       resourceDeletionQueue;

    vk::raii::Context                           context {};
    std::unique_ptr<vk::raii::Instance>         instance;
    vk::raii::SurfaceKHR                        surface { nullptr };
    Window::Extent                              windowExtent {};

    vk::raii::DebugUtilsMessengerEXT            debugMessenger { nullptr };

    std::unique_ptr<vk::raii::PhysicalDevice>   physicalDevice {};
    vk::raii::Device                            device { nullptr };

    MemoryAllocator     memoryAllocator {};

    QueueFamilyIndices  queueFamilyIndices {};
    uint32_t        graphicsIndex {};
    uint32_t        presentIndex {};
    uint32_t        transferIndex {};
    vk::raii::Queue graphicsQueue { nullptr };
    vk::raii::Queue presentQueue { nullptr };
    vk::raii::Queue transferQueue { nullptr };

    vk::SampleCountFlagBits msaaSamples { vk::SampleCountFlagBits::e1 };

    vk::raii::SwapchainKHR              swapChain { nullptr };
    std::vector<vk::Image>              swapChainImages;
    std::vector<vk::raii::ImageView>    swapChainImageViews;
    vk::Format                          swapChainImageFormat {};
    vk::Extent2D                        swapChainExtent;
    bool                                swapChainOutOfDate {false};

    // Traditional render pass (fallback for non-dynamic rendering)
    vk::raii::RenderPass                renderPass { nullptr };
    std::vector<vk::raii::Framebuffer>  swapChainFramebuffers;

    DescriptorAllocator                 globalDescriptorAllocator {};
    vk::raii::DescriptorSetLayout       drawImageDescriptorSetLayout { nullptr };

    vk::raii::PipelineLayout            pipelineLayout { nullptr };
    vk::raii::Pipeline                  graphicsPipeline { nullptr };

    // Need more than one draw image for each frame in flight
    AllocatedImage  drawImage;
    AllocatedImage  colorImage;
    AllocatedImage  depthImage;

    uint32_t                mipLevels {0};
    ktxVulkanTexture        texture;
    vk::raii::ImageView     textureImageView { nullptr };
    vk::raii::Sampler       textureSampler { nullptr };

    std::vector<Vertex>     vertices {};
    std::vector<uint32_t>   indices {};
    // vk::raii::Buffer        vertexBuffer { nullptr };
    // vk::raii::DeviceMemory  vertexBufferMemory { nullptr };
    // vk::raii::Buffer        indexBuffer { nullptr };
    // vk::raii::DeviceMemory  indexBufferMemory { nullptr };
    AllocatedBuffer     vertexBuffer;
    AllocatedBuffer     indexBuffer;

    vk::raii::CommandPool                   graphicsCommandPool { nullptr };
    vk::raii::CommandPool                   transferCommandPool { nullptr };

    std::vector<FrameData>      frames {};
    size_t      currentFrame {0};

    std::array<ModelObject, MAX_OBJECTS>        modelObjects {};

    const std::string TEXTURE_PATH {"models/viking_room/textures/viking_room.ktx2"};
    float minX { std::numeric_limits<float>::max() };
    float maxX { std::numeric_limits<float>::min() };
    float minY { minX };
    float maxY { maxX };
    float minZ { minX };
    float maxZ { maxX };

    std::vector<const char*> requiredDeviceExtensions = {
    #ifdef __APPLE__
        "VK_KHR_portability_subset",
    #endif
        vk::KHRSwapchainExtensionName,
        vk::KHRSpirv14ExtensionName,
        vk::KHRCreateRenderpass2ExtensionName,
        vk::KHRCopyCommands2ExtensionName
    };

private:
    void cleanup();
    void cleanupSwapChain();

    void createInstance();
    void setupDebugMessenger();
    void createSurface();
    void pickPhysicalDevice();
    void checkFeatureSupport();
    void detectFeatureSupport();
    void createLogicalDevice();
    void createMemoryAllocator();
    void createSwapChain();

    void initDescriptors();
    void createGraphicsPipeline();

    void createCommandPool();

    void createColorResources();
    void createDepthResources();
    void createRenderPass();
    void createFramebuffers();

    void createTextureImage();
    void createTextureImageView();
    void createTextureSampler();

    void loadModel(const std::string& modelName);
    void setupModelObjects();
    void createVertexBuffer();
    void createIndexBuffer();
    void createUniformBuffers();
    void createDescriptorSets();

    void createFrameData();


    void updateUniformBuffer(uint32_t currentImage);
    void drawFrame();
    void drawBackground(const vk::raii::CommandBuffer& commandBuffer, uint32_t imageIndex);

    /**
     * Helper functions
     */
    bool isDeviceSuitable(const vk::raii::PhysicalDevice& device);
    [[nodiscard]] std::vector<const char*> getRequiredLayers() const;
    void checkLayerSupport(const std::vector<const char*>& requiredLayers) const;
    [[nodiscard]] std::vector<const char*> getRequiredExtensions() const;
    void checkExtensionSupport(const std::vector<const char*>& requiredExtensions) const;
    uint32_t rateDeviceSuitability(const vk::raii::PhysicalDevice& device);
    [[nodiscard]] QueueFamilyIndices findQueueFamilies(const vk::raii::PhysicalDevice& physicalDevice) const;
    [[nodiscard]] uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) const;
    [[nodiscard]] vk::Format findSupportedFormat(
        const std::vector<vk::Format>& candidates,
        vk::ImageTiling tiling,
        vk::FormatFeatureFlags features
    ) const;
    [[nodiscard]] vk::Format findDepthFormat() const;
    [[nodiscard]] vk::SampleCountFlagBits getMaxUsableSampleCount() const;
    [[nodiscard]] vk::Format chooseSwapSurfaceFormat(
        const std::vector<vk::SurfaceFormatKHR>& availableFormats) const noexcept;
    [[nodiscard]] vk::PresentModeKHR chooseSwapPresentMode(
        const std::vector<vk::PresentModeKHR>& availablePresentModes
    ) const noexcept;
    [[nodiscard]]
    vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities) const noexcept;

    void createBuffer(
        vk::DeviceSize size,
        vk::BufferUsageFlags usage,
        vk::MemoryPropertyFlags properties,
        vk::raii::Buffer& buffer,
        vk::raii::DeviceMemory& bufferMemory
    ) const;

    void copyBuffer(
        const vk::Buffer& srcBuffer,
        const vk::Buffer& dstBuffer,
        vk::DeviceSize size
    ) const;

    [[nodiscard]]
    vk::raii::CommandBuffer beginSingleTimeCommands(const vk::raii::CommandPool& commandPool) const;
    void endSingleTimeCommands(
        const vk::raii::CommandBuffer& commandBuffer,
        const vk::raii::Queue& queue
    ) const;
    
    void copyBufferToImage(
        const vk::raii::Buffer& buffer,
        const vk::raii::Image& image,
        uint32_t width,
        uint32_t height
    ) const;
    void generateMipmaps(
        const vk::raii::Image& image,
        vk::Format imageFormat,
        int32_t texWidth,
        int32_t texHeight,
        uint32_t mipLevels
    ) const;
};