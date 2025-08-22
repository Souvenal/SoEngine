#pragma once

#include "config.h"
#include "vertex.h"
#include "descriptor.h"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <stb_image.h>

#include <tiny_obj_loader.h>

#include <limits>       // std::numeric_limits
#include <vector>
#include <memory>
#include <filesystem>

struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily; // support drawing commands
    std::optional<uint32_t> presentFamily;  // support presentation
    std::optional<uint32_t> transferFamily; // support transfer operations

    [[nodiscard]] bool isComplete() const {
        return
            graphicsFamily.has_value() &&
            presentFamily.has_value() &&
            transferFamily.has_value();
    }
};

class Application {
public:
    explicit Application(const std::filesystem::path& path);

    void run();

private:
    AppInfo appInfo;
    std::filesystem::path appDir;

    const std::string TEXTURE_PATH {"models/viking_room/textures/viking_room.png"};
    tinyobj::real_t minX { std::numeric_limits<tinyobj::real_t>::max() };
    tinyobj::real_t maxX { std::numeric_limits<tinyobj::real_t>::min() };
    tinyobj::real_t minY { minX };
    tinyobj::real_t maxY { maxX };
    tinyobj::real_t minZ { minX };
    tinyobj::real_t maxZ { maxX };

    GLFWwindow*                                 window { nullptr };
    vk::raii::Context                           context {};
    std::unique_ptr<vk::raii::Instance>         instance;
    vk::raii::SurfaceKHR                        surface { nullptr };

    vk::raii::DebugUtilsMessengerEXT            debugMessenger { nullptr };

    std::unique_ptr<vk::raii::PhysicalDevice>   physicalDevice {};
    vk::raii::Device                            device { nullptr };
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
    bool                                framebufferResized {false};;

    // Traditional render pass (fallback for non-dynamic rendering)
    vk::raii::RenderPass                renderPass { nullptr };
    std::vector<vk::raii::Framebuffer>  swapChainFramebuffers;

    vk::raii::Image                 colorImage { nullptr };
    vk::raii::DeviceMemory          colorImageMemory { nullptr };
    vk::raii::ImageView             colorImageView { nullptr };

    vk::raii::Image         depthImage { nullptr };
    vk::raii::DeviceMemory  depthImageMemory { nullptr };
    vk::raii::ImageView     depthImageView { nullptr };

    uint32_t                mipLevels {0};
    vk::raii::Image         textureImage { nullptr };
    vk::raii::DeviceMemory  textureImageMemory { nullptr };
    vk::raii::ImageView     textureImageView { nullptr };
    vk::raii::Sampler       textureSampler { nullptr };

    std::vector<Vertex>     vertices;
    std::vector<uint32_t>   indices;
    vk::raii::Buffer        vertexBuffer { nullptr };
    vk::raii::DeviceMemory  vertexBufferMemory { nullptr };
    vk::raii::Buffer        indexBuffer { nullptr };
    vk::raii::DeviceMemory  indexBufferMemory { nullptr };

    std::vector<vk::raii::Buffer>       uniformBuffers {};
    std::vector<vk::raii::DeviceMemory> uniformBuffersMemory {};
    std::vector<void*>                  uniformBuffersMapped {};

    vk::raii::DescriptorSetLayout           descriptorSetLayout { nullptr };
    vk::raii::DescriptorPool                descriptorPool { nullptr };
    std::vector<vk::raii::DescriptorSet>    descriptorSets {};

    vk::raii::PipelineLayout                pipelineLayout { nullptr };
    vk::raii::Pipeline                      graphicsPipeline { nullptr };

    vk::raii::CommandPool                   graphicsCommandPool { nullptr };
    std::vector<vk::raii::CommandBuffer>    graphicsCommandBuffers {};
    vk::raii::CommandPool                   transferCommandPool { nullptr };

    std::vector<vk::raii::Semaphore>    presentCompleteSemaphores;
    std::vector<vk::raii::Semaphore>    renderFinishedSemaphores;
    std::vector<vk::raii::Fence>        inFlightFences;
    size_t      currentFrame {0};

    std::vector<const char*> requiredDeviceExtensions = {
    #ifdef __APPLE__
        "VK_KHR_portability_subset",
    #endif
        vk::KHRSwapchainExtensionName,
        vk::KHRSpirv14ExtensionName,
        vk::KHRCreateRenderpass2ExtensionName
    };

private:
    void initWindow();
    void initVulkan();
    void mainLoop();
    void cleanup();
    void cleanupSwapChain();

    void createInstance();
    void setupDebugMessenger();
    void createSurface();
    void pickPhysicalDevice();
    void detectFeatureSupport();
    void createLogicalDevice();

    void createSwapChain();
    void recreateSwapChain();
    void createImageViews();

    void createDescriptorSetLayout();
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
    void createVertexBuffer();
    void createIndexBuffer();
    void createUniformBuffers();

    void createDescriptorPool();
    void createDescriptorSets();
    void createCommandBuffers();
    void createSyncObjects();

    void recordCommandBuffer(uint32_t imageIndex) const;
    void updateUniformBuffer(uint32_t currentImage) const;
    void drawFrame();

    /**
     * Helper functions
     */
    static void framebufferResizeCallback(GLFWwindow* window, int width, int height);
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

    [[nodiscard]]
    vk::raii::ShaderModule createShaderModule(const std::vector<char>& code) const;

    void createBuffer(
        vk::DeviceSize size,
        vk::BufferUsageFlags usage,
        vk::MemoryPropertyFlags properties,
        vk::raii::Buffer& buffer,
        vk::raii::DeviceMemory& bufferMemory
    ) const;

    void createImage(
        uint32_t width,
        uint32_t height,
        uint32_t mipLevels,
        vk::SampleCountFlagBits numSamples,
        vk::Format format,
        vk::ImageTiling tiling,
        vk::ImageUsageFlags usage,
        vk::MemoryPropertyFlags properties,
        vk::raii::Image& image,
        vk::raii::DeviceMemory& imageMemory
    ) const;

    [[nodiscard]]
    vk::raii::ImageView createImageView(
        const vk::raii::Image& image,
        vk::Format format,
        vk::ImageAspectFlags aspectFlags,
        uint32_t mipLevels
    ) const;

    [[nodiscard]]
    std::unique_ptr<vk::raii::CommandBuffer> beginSingleTimeCommands(const vk::raii::CommandPool& commandPool) const;
    void endSingleTimeCommands(
        const vk::raii::CommandBuffer& commandBuffer,
        const vk::raii::Queue& queue
    ) const;

    void copyBuffer(
        const vk::raii::Buffer& srcBuffer,
        const vk::raii::Buffer& dstBuffer,
        vk::DeviceSize size
    ) const;

    void transitionImageLayout(
        const vk::raii::Image& image,
        // vk::Format format,
        vk::ImageLayout oldLayout,
        vk::ImageLayout newLayout,
        uint32_t mipLevels
    ) const;
    void transitionImageLayout(
        const vk::Image& image,
        // vk::Format format,
        vk::ImageLayout oldLayout,
        vk::ImageLayout newLayout,
        uint32_t mipLevels
    ) const;
    void transitionImageLayout(
        const vk::raii::Image& image,
        vk::ImageLayout oldLayout,
        vk::ImageLayout newLayout,
        vk::Flags<vk::AccessFlagBits2> srcAccessMask,
        vk::Flags<vk::AccessFlagBits2> dstAccessMask,
        vk::PipelineStageFlagBits2 srcStageMask,
        vk::PipelineStageFlagBits2 dstStageMask,
        vk::ImageAspectFlagBits aspectMask,
        uint32_t mipLevels
    ) const;
    void transitionImageLayout(
        const vk::Image& image,
        vk::ImageLayout oldLayout,
        vk::ImageLayout newLayout,
        vk::Flags<vk::AccessFlagBits2> srcAccessMask,
        vk::Flags<vk::AccessFlagBits2> dstAccessMask,
        vk::PipelineStageFlagBits2 srcStageMask,
        vk::PipelineStageFlagBits2 dstStageMask,
        vk::ImageAspectFlagBits aspectMask,
        uint32_t mipLevels
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