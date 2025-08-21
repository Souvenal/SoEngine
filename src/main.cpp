#include "base.h"
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

#include <cstdint>      // uint32_t
#include <limits>       // std::numeric_limits
#include <algorithm>    // std::clamp
#include <iostream>
#include <vector>
#include <set>
#include <map>
#include <unordered_map>
#include <stdexcept>
#include <cstdlib>
#include <memory>
#include <filesystem>
#include <fstream>
#include <chrono>

constexpr uint32_t WIDTH {800};
constexpr uint32_t HEIGHT {600};
constexpr size_t MAX_FRAMES_IN_FLIGHT { 3 };

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    vk::DebugUtilsMessageSeverityFlagBitsEXT severity,
    vk::DebugUtilsMessageTypeFlagsEXT messageType,
    const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData) {
    if (severity >= vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning) {
        std::println(std::cerr, "Validation layer: type {} msg: {}\n", to_string(messageType), pCallbackData->pMessage);
    }
    // indicate if the call that triggered this callback should be aborted
    return vk::False;
}

static std::vector<char> readFile(const std::string& filename) {
    std::ifstream file (filename, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file!");
    }

    const size_t fileSize = static_cast<size_t>(file.tellg());
    std::vector<char> buffer(fileSize);

    file.seekg(0, std::ios::beg);
    file.read(buffer.data(), static_cast<std::streamsize>(fileSize));
    file.close();

    return buffer;
}

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
    explicit Application(const std::filesystem::path&& path): appDir(path) {}

    void run() {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

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
    void initWindow() {
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

        window = glfwCreateWindow(static_cast<int>(WIDTH), static_cast<int>(HEIGHT), "Vulkan", nullptr, nullptr);
        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
    }

    void initVulkan() {
        createInstance();
        setupDebugMessenger();
        createSurface();        // influence physical device selection
        pickPhysicalDevice();
        detectFeatureSupport();
        createLogicalDevice();
        createSwapChain();
        createImageViews();

        if (!appInfo.dynamicRenderingSupported) {
            createRenderPass();
        }
        createDescriptorSetLayout();
        createGraphicsPipeline();
        createCommandPool();

        createColorResources();
        createDepthResources();
        if (!appInfo.dynamicRenderingSupported) {
            createFramebuffers();
        }
        createTextureImage();
        createTextureImageView();
        createTextureSampler();

        // loadModel("amiya-arknights");
        loadModel("viking_room");
        createVertexBuffer();
        createIndexBuffer();

        createUniformBuffers();
        createDescriptorPool();
        createDescriptorSets();

        createCommandBuffers();
        createSyncObjects();

        // Print feature support summary
        appInfo.printFeatureSupportSummary();
    }

    void mainLoop() {
        while(!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            drawFrame();
        }

        device.waitIdle();
    }

    void cleanup() {
        cleanupSwapChain();

        glfwDestroyWindow(window);
        glfwTerminate();
    }

    void cleanupSwapChain() {
        swapChainImageViews.clear();
        swapChain = nullptr;
    }

    void createInstance() {
        constexpr vk::ApplicationInfo appInfo {
            .pApplicationName = "Hello Triangle",
            .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
            .pEngineName = "No Engine",
            .engineVersion = VK_MAKE_VERSION(1, 0, 0),
            .apiVersion = vk::ApiVersion14
        };

        auto requiredLayers = getRequiredLayers();
        auto requiredExtensions = getRequiredExtensions();

        vk::InstanceCreateInfo createInfo {
            .pApplicationInfo = &appInfo,
            .enabledLayerCount = static_cast<uint32_t>(requiredLayers.size()),
            .ppEnabledLayerNames = requiredLayers.data(),
            .enabledExtensionCount = static_cast<uint32_t>(requiredExtensions.size()),
            .ppEnabledExtensionNames = requiredExtensions.data()
        };
#ifdef __APPLE__
        // createInfo.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
        createInfo.flags |= vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR;
#endif

        instance = std::make_unique<vk::raii::Instance>(context, createInfo);
    }

    void setupDebugMessenger() {
        // Only used if validation layers are enabled via vkconfig
        vk::DebugUtilsMessageSeverityFlagsEXT severityFlags {
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eError
        };
        vk::DebugUtilsMessageTypeFlagsEXT messageTypeFlags {
            vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
            vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance |
            vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation
        };
        vk::DebugUtilsMessengerCreateInfoEXT debugUtilsMessengerCreateInfoEXT {
            .messageSeverity = severityFlags,
            .messageType = messageTypeFlags,
            .pfnUserCallback = &debugCallback
        };
        try {
            debugMessenger = instance->createDebugUtilsMessengerEXT(debugUtilsMessengerCreateInfoEXT);
        } catch (const vk::SystemError& err) {
            std::println("Debug messenger not available. Validation layers may not be enabled.");
        }
    }

    void createSurface() {
        VkSurfaceKHR _surface;
        if (glfwCreateWindowSurface(**instance, window, nullptr, &_surface) != 0) {
            throw std::runtime_error("Failed to create window surface!");
        }
        surface = vk::raii::SurfaceKHR(*instance, _surface);
    }

    void pickPhysicalDevice() {
        auto devices = instance->enumeratePhysicalDevices();
        if (devices.empty()) {
            throw std::runtime_error("Failed to GPUs with Vulkan support!");
        }

        // Use an ordered map to automatically sort candidates by increasing score
        std::multimap<uint32_t, vk::raii::PhysicalDevice> candidates;
        for (const auto& device : devices) {
            uint32_t score = rateDeviceSuitability(device);
            candidates.insert(std::make_pair(score, device));
        }

        // Check if the best candidate is suitable at all
        if (candidates.rbegin()->first == 0) {
            throw std::runtime_error("Failed to find a suitable GPU!");
        }
        
        physicalDevice = std::make_unique<vk::raii::PhysicalDevice>(candidates.rbegin()->second);
        msaaSamples = getMaxUsableSampleCount();
        std::println("Physical GPU score: {}", candidates.rbegin()->first);
    }

    void detectFeatureSupport() {
        auto deviceProperties = physicalDevice->getProperties();
        auto availableExtensions = physicalDevice->enumerateDeviceExtensionProperties();

        // Check for dynamic rendering support
        if (deviceProperties.apiVersion >= vk::ApiVersion13) {
            appInfo.dynamicRenderingSupported = true;
            std::println("Dynamic rendering supported via Vulkan 1.3");
        } else if (std::ranges::any_of(availableExtensions,
            [](const auto& availableExtension) {
                return strcmp(availableExtension.extensionName, vk::KHRDynamicRenderingExtensionName) == 0;
            })) {
            appInfo.dynamicRenderingSupported = true;
            std::println("Dynamic rendering supported via extension");
        }

        // Check for timeline semaphores support
        if (deviceProperties.apiVersion >= vk::ApiVersion12) {
            appInfo.timelineSemaphoresSupported = true;
            std::println("Timeline semaphores supported via Vulkan 1.2");
        } else if (std::ranges::any_of(availableExtensions,
            [](const auto& availableExtension) {
                return strcmp(availableExtension.extensionName, vk::KHRTimelineSemaphoreExtensionName) == 0;
            })) {
            appInfo.timelineSemaphoresSupported = true;
            std::println("Timeline semaphores supported via extension");
        }

        // Check for synchronization2 support
        if (deviceProperties.apiVersion >= vk::ApiVersion13) {
            appInfo.synchronization2Supported = true;
            std::println("Synchronization2 supported via Vulkan 1.3");
        } else if (std::ranges::any_of(availableExtensions,
            [](const auto& availableExtension) {
                return strcmp(availableExtension.extensionName, vk::KHRSynchronization2ExtensionName) == 0;
            })) {
            appInfo.synchronization2Supported = true;
            std::println("Synchronization2 supported via extension");
        }

        if (appInfo.dynamicRenderingSupported && deviceProperties.apiVersion < vk::ApiVersion13) {
            requiredDeviceExtensions.emplace_back(vk::KHRDynamicRenderingExtensionName);
        }
        if (appInfo.timelineSemaphoresSupported && deviceProperties.apiVersion < vk::ApiVersion12) {
            requiredDeviceExtensions.emplace_back(vk::KHRTimelineSemaphoreExtensionName);
        }
        if (appInfo.synchronization2Supported && deviceProperties.apiVersion < vk::ApiVersion13) {
            requiredDeviceExtensions.emplace_back(vk::KHRSynchronization2ExtensionName);
        }
    }

    void createLogicalDevice() {
        const auto queueFamilyProperties = physicalDevice->getQueueFamilyProperties();
        const auto indices = findQueueFamilies(*physicalDevice);
        graphicsIndex = indices.graphicsFamily.value();
        presentIndex = indices.presentFamily.value();
        transferIndex = indices.transferFamily.value();

        std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
        std::set<uint32_t> uniqueQueueFamilies = {
            graphicsIndex, presentIndex, transferIndex
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

        vk::StructureChain<
            vk::PhysicalDeviceFeatures2,
            vk::PhysicalDeviceVulkan13Features,
            vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT
        > featureChain {
            {.features = { .sampleRateShading = vk::True, .samplerAnisotropy = vk::True}},
            {.synchronization2 =  vk::True, .dynamicRendering = true},
            {.extendedDynamicState = true}
        };

        vk::DeviceCreateInfo deviceCreateInfo {
            .pNext = &featureChain.get<vk::PhysicalDeviceFeatures2>(),
            .queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size()),
            .pQueueCreateInfos = queueCreateInfos.data(),
            .enabledExtensionCount = static_cast<uint32_t>(requiredDeviceExtensions.size()),
            .ppEnabledExtensionNames = requiredDeviceExtensions.data(),
        };
        device = vk::raii::Device( *physicalDevice, deviceCreateInfo );

        graphicsQueue = vk::raii::Queue(device, graphicsIndex, 0);
        presentQueue = vk::raii::Queue(device, presentIndex, 0);
        transferQueue = vk::raii::Queue(device, transferIndex, 0);
    }

    void createSwapChain() {
        auto surfaceCapabilities = physicalDevice->getSurfaceCapabilitiesKHR(*surface);
        swapChainImageFormat = chooseSwapSurfaceFormat(
            physicalDevice->getSurfaceFormatsKHR(*surface)
        );
        swapChainExtent = chooseSwapExtent(surfaceCapabilities);
        auto minImageCount = std::max( 3u, surfaceCapabilities.minImageCount );
        minImageCount = ( surfaceCapabilities.maxImageCount > 0 && minImageCount > surfaceCapabilities.maxImageCount ) ?
            surfaceCapabilities.maxImageCount : minImageCount;


        vk::SwapchainCreateInfoKHR createInfo {
            .surface = surface,
            .minImageCount = minImageCount,
            .imageFormat = swapChainImageFormat,
            .imageColorSpace = vk::ColorSpaceKHR::eSrgbNonlinear,
            .imageExtent = swapChainExtent,
            .imageArrayLayers = 1,  // the amount of layers each image consists of, always 1 unless developing a stereoscopic 3D app
            .imageUsage = vk::ImageUsageFlagBits::eColorAttachment,   // operations using the images in the swap chain for
            .preTransform = surfaceCapabilities.currentTransform,
            .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
            .presentMode = chooseSwapPresentMode(physicalDevice->getSurfacePresentModesKHR(*surface)),
            .clipped = vk::True,
            .oldSwapchain = nullptr // specify one when the window is resized,
        };

        // Specify how to handle swap chain images across multiple queue families
        uint32_t queueFamilyIndices[] = {
            graphicsIndex, presentIndex
        };

        if (graphicsIndex == presentIndex) {
            createInfo.imageSharingMode = vk::SharingMode::eExclusive;
            createInfo.queueFamilyIndexCount = 0;
            createInfo.pQueueFamilyIndices = nullptr;
        } else {
            createInfo.imageSharingMode = vk::SharingMode::eConcurrent,
            createInfo.queueFamilyIndexCount = 2,
            createInfo.pQueueFamilyIndices = queueFamilyIndices;
        }

        swapChain = vk::raii::SwapchainKHR(device, createInfo);
        swapChainImages = swapChain.getImages();

        std::println("Number of images in the swap chain: {}", swapChainImages.size());
    }

    void createImageViews() {
        vk::ImageViewCreateInfo imageViewCreateInfo{
            .viewType = vk::ImageViewType::e2D,
            .format = swapChainImageFormat,
            .subresourceRange = {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1
            }
        };
        for (const auto& image : swapChainImages) {
            imageViewCreateInfo.image = image;
            swapChainImageViews.emplace_back(device, imageViewCreateInfo);
        }
    }

    void createRenderPass() {
        if (appInfo.dynamicRenderingSupported) {
            std::println("Using dynamic rendering, skipping render pass creation.");
            return;
        }

        std::println("Creating traditional render pass.");

        vk::AttachmentDescription colorAttachment {
            .format = swapChainImageFormat,
            .samples = msaaSamples,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
            .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
            .initialLayout = vk::ImageLayout::eUndefined,
            .finalLayout = vk::ImageLayout::eColorAttachmentOptimal,
        };

        vk::AttachmentReference colorAttachmentRef {
            .attachment = 0,
            .layout = vk::ImageLayout::eColorAttachmentOptimal
        };

        vk::AttachmentDescription depthAttachment {
            .format = findDepthFormat(),
            .samples = msaaSamples,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eDontCare,
            .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
            .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
            .initialLayout = vk::ImageLayout::eUndefined,
            .finalLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal
        };

        vk::AttachmentReference depthAttachmentRef {
            .attachment = 1,
            .layout = vk::ImageLayout::eDepthStencilAttachmentOptimal
        };

        vk::AttachmentDescription colorAttachmentResolve {
            .format = swapChainImageFormat,
            .samples = vk::SampleCountFlagBits::e1,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
            .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
            .initialLayout = vk::ImageLayout::eUndefined,
            .finalLayout = vk::ImageLayout::ePresentSrcKHR
        };


        vk::AttachmentReference colorAttachmentResolveRef {
            .attachment = 2,
            .layout = vk::ImageLayout::eColorAttachmentOptimal,
        };

        std::array attachments {
            colorAttachment, depthAttachment, colorAttachmentResolve
        };

        vk::SubpassDescription subpass {
            .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
            .colorAttachmentCount = 1,
            .pColorAttachments = &colorAttachmentRef,
            .pResolveAttachments = &colorAttachmentResolveRef,
            .pDepthStencilAttachment = &depthAttachmentRef
        };

        vk::SubpassDependency dependency {
            .srcSubpass = vk::SubpassExternal,
            .dstSubpass = 0,
            .srcStageMask =
                vk::PipelineStageFlagBits::eColorAttachmentOutput |
                vk::PipelineStageFlagBits::eEarlyFragmentTests,
            .dstStageMask =
                vk::PipelineStageFlagBits::eColorAttachmentOutput |
                vk::PipelineStageFlagBits::eEarlyFragmentTests,
            .srcAccessMask = {},
            .dstAccessMask =
                vk::AccessFlagBits::eColorAttachmentWrite |
                vk::AccessFlagBits::eDepthStencilAttachmentWrite,
        };

        vk::RenderPassCreateInfo renderPassInfo {
            .attachmentCount = static_cast<uint32_t>(attachments.size()),
            .pAttachments = attachments.data(),
            .subpassCount = 1,
            .pSubpasses = &subpass,
            .dependencyCount = 1,
            .pDependencies = &dependency
        };

        renderPass = vk::raii::RenderPass{device, renderPassInfo};
    }

    void createFramebuffers() {
        if (appInfo.dynamicRenderingSupported) {
            // No framebuffers needed with dynamic rendering
            std::println("Using dynamic rendering, skipping framebuffer creation.");
            return;
        }

        std::println("Creating traditional framebuffers");

        swapChainFramebuffers.clear();
        for (size_t i = 0; i < swapChainImageViews.size(); ++i) {
            std::array attachments {
                *colorImageView,
                *depthImageView,
                *swapChainImageViews[i]
            };
        
            vk::FramebufferCreateInfo framebufferInfo {
                .renderPass = *renderPass,
                .attachmentCount = static_cast<uint32_t>(attachments.size()),
                .pAttachments = attachments.data(),
                .width = swapChainExtent.width,
                .height = swapChainExtent.height,
                .layers = 1
            };
        
            swapChainFramebuffers.emplace_back(device, framebufferInfo);
        }
    }

    void createDescriptorSetLayout() {
        vk::DescriptorSetLayoutBinding uboLayoutBinding {
            0,
            vk::DescriptorType::eUniformBuffer,
            1,
            vk::ShaderStageFlagBits::eVertex,
            nullptr
        };

        vk::DescriptorSetLayoutBinding samplerLayoutBinding {
            .binding = 1,
            .descriptorType = vk::DescriptorType::eCombinedImageSampler,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eFragment,
            .pImmutableSamplers = nullptr,
        };

        std::array bindings {
            uboLayoutBinding, samplerLayoutBinding
        };

        vk::DescriptorSetLayoutCreateInfo layoutInfo {
            .bindingCount = static_cast<uint32_t>(bindings.size()),
            .pBindings = bindings.data()
        };
        descriptorSetLayout = vk::raii::DescriptorSetLayout{device, layoutInfo};
    }

    void createGraphicsPipeline() {
        auto shadersDir = appDir / "shaders";

        std::string shaderPath = (shadersDir / "slang.spv").string();
        const auto shaderModule = createShaderModule(readFile(shaderPath));

        // Shader stage creation
        vk::PipelineShaderStageCreateInfo vertShaderStageInfo {
            .stage = vk::ShaderStageFlagBits::eVertex,
            .module = shaderModule,
            .pName = "vertMain"
        };
        vk::PipelineShaderStageCreateInfo fragShaderStageInfo {
            .stage = vk::ShaderStageFlagBits::eFragment,
            .module = shaderModule,
            .pName = "fragMain"
        };

        vk::PipelineShaderStageCreateInfo shaderStages[] = {
            vertShaderStageInfo, fragShaderStageInfo
        };

        // Vertex input
        auto bindingDescription = Vertex::getBindingDescription();
        auto attributeDescriptions = Vertex::getAttributeDescriptions();
        vk::PipelineVertexInputStateCreateInfo vertexInputInfo {
            .vertexBindingDescriptionCount = 1,
            .pVertexBindingDescriptions = &bindingDescription,
            .vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size()),
            .pVertexAttributeDescriptions = attributeDescriptions.data()
        };

        // dynamic state
        std::vector<vk::DynamicState> dynamicStates {
            vk::DynamicState::eViewport,
            vk::DynamicState::eScissor,
        };
        vk::PipelineDynamicStateCreateInfo dynamicState {
            .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
            .pDynamicStates = dynamicStates.data()
        };

        vk::PipelineInputAssemblyStateCreateInfo inputAssembly {
            .topology = vk::PrimitiveTopology::eTriangleList,
            .primitiveRestartEnable = VK_FALSE
        };

        // Viewports and scissors
        // static
        // VkViewport viewport {
        //     .x = 0.0f,
        //     .y = 0.0f,
        //     .width = static_cast<float>(swapChainExtent.width),
        //     .height = static_cast<float>(swapChainExtent.height),
        //     .minDepth = 0.0f,
        //     .maxDepth = 1.0f
        // };
        // VkRect2D scissor {
        //     .offset = {0, 0},
        //     .extent = swapChainExtent
        // };
        vk::PipelineViewportStateCreateInfo viewportSate {
            .viewportCount = 1,
            // .pViewports = &viewport,
            .scissorCount = 1,
            // .pScissors = &scissor
        };

        // Rasterization
        vk::PipelineRasterizationStateCreateInfo rasterizer {
            .depthClampEnable = vk::False,
            .rasterizerDiscardEnable = vk::False,
            .polygonMode = vk::PolygonMode::eFill,
            // .polygonMode = VK_POLYGON_MODE_LINE,
            .cullMode = vk::CullModeFlagBits::eBack,
            // .frontFace = VK_FRONT_FACE_CLOCKWISE,
            .frontFace = vk::FrontFace::eCounterClockwise, // Y-flip
            .depthBiasEnable = vk::False, // use for the shadow map
            .depthBiasConstantFactor = 0.0f, // Optional
            .depthBiasClamp = 0.0f, //Optional
            .depthBiasSlopeFactor = 0.0f, // Optional
            .lineWidth = 1.0f
        };

        // Multisampling
        vk::PipelineMultisampleStateCreateInfo multisampling {
            .rasterizationSamples = msaaSamples,
            .sampleShadingEnable = vk::True,
            .minSampleShading = 0.2f, // min fraction for sample shading; closer one is smoother
            .pSampleMask = nullptr, // Optional
            .alphaToCoverageEnable = vk::False, // Optional
            .alphaToOneEnable = vk::False // Optional
        };

        // Depth and stencil testing
        vk::PipelineDepthStencilStateCreateInfo depthStencil {
            .depthTestEnable = vk::True,
            .depthWriteEnable = vk::True,
            .depthCompareOp = vk::CompareOp::eLess,
            .depthBoundsTestEnable = vk::False,
            .stencilTestEnable = vk::False,
            .front = {},    // Optional
            .back = {},     // Optional
            .minDepthBounds = 0.0f, // Optional
            .maxDepthBounds = 1.0f, // Optional
        };

        // Color blending
        vk::PipelineColorBlendAttachmentState colorBlendAttachment {
            .blendEnable = vk::False,
            .srcColorBlendFactor = vk::BlendFactor::eOne, // Optional
            .dstColorBlendFactor = vk::BlendFactor::eZero, // Optional
            .colorBlendOp = vk::BlendOp::eAdd, // Optional
            .srcAlphaBlendFactor = vk::BlendFactor::eOne, // Optional
            .dstAlphaBlendFactor = vk::BlendFactor::eZero, // Optional
            .alphaBlendOp = vk::BlendOp::eAdd, // Optional
            .colorWriteMask =
                vk::ColorComponentFlagBits::eR |
                vk::ColorComponentFlagBits::eG |
                vk::ColorComponentFlagBits::eB |
                vk::ColorComponentFlagBits::eA
        }; // configuration per attached framebuffer
        // Alpha blending
        // VkPipelineColorBlendAttachmentState colorBlendAttachment {
        //     .blendEnable = VK_TRUE,
        //     .srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA, // Optional
        //     .dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA, // Optional
        //     .colorBlendOp = VK_BLEND_OP_ADD, // Optional
        //     .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE, // Optional
        //     .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO, // Optional
        //     .alphaBlendOp = VK_BLEND_OP_ADD, // Optional
        //     .colorWriteMask =
        //         VK_COLOR_COMPONENT_R_BIT |
        //         VK_COLOR_COMPONENT_G_BIT |
        //         VK_COLOR_COMPONENT_B_BIT |
        //         VK_COLOR_COMPONENT_A_BIT
        // };

        vk::PipelineColorBlendStateCreateInfo colorBlending {
            .logicOpEnable = vk::False,
            .logicOp = vk::LogicOp::eCopy, // Optional
            .attachmentCount = 1,
            .pAttachments = &colorBlendAttachment,
        };

        // Pipeline layout
        vk::PipelineLayoutCreateInfo pipelineLayoutInfo {
            .setLayoutCount = 1,
            .pSetLayouts = &*descriptorSetLayout,
            .pushConstantRangeCount = 0, // Optional
            .pPushConstantRanges = nullptr // Optional
        };
        pipelineLayout = vk::raii::PipelineLayout{device, pipelineLayoutInfo};

        vk::PipelineRenderingCreateInfo pipelineRenderingCreateInfo {
            .colorAttachmentCount = 1,
            .pColorAttachmentFormats = &swapChainImageFormat,
            .depthAttachmentFormat = findDepthFormat(),
        };

        vk::GraphicsPipelineCreateInfo pipelineInfo {
            .stageCount = 2,
            .pStages = shaderStages,
            .pVertexInputState = &vertexInputInfo,
            .pInputAssemblyState = &inputAssembly,
            .pViewportState = &viewportSate,
            .pRasterizationState = &rasterizer,
            .pMultisampleState = &multisampling,
            .pDepthStencilState = &depthStencil, // Optional
            .pColorBlendState = &colorBlending,
            .pDynamicState = &dynamicState,
            .layout = pipelineLayout,
            // .renderPass = renderPass,
            .renderPass = nullptr,  // Dynamic rendering
            .basePipelineHandle = VK_NULL_HANDLE, // Optional
            .basePipelineIndex = -1 // Optional
        };

        // Dynamic rendering
        if (appInfo.dynamicRenderingSupported) {
            std::println("Configuring pipeline for dynamic rendering");
            pipelineInfo.pNext = &pipelineRenderingCreateInfo,
            pipelineInfo.renderPass = nullptr;
        } else {
            std::println("Configuring pipeline for traditional render pass");
            pipelineInfo.pNext = nullptr;
            pipelineInfo.renderPass = *renderPass;
            pipelineInfo.subpass = 0;
        }

        graphicsPipeline = vk::raii::Pipeline{device, nullptr, pipelineInfo};
    }

    void createCommandPool() {
        vk::CommandPoolCreateInfo graphicsCommandPoolInfo {
            .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
            .queueFamilyIndex = graphicsIndex
        };
        graphicsCommandPool = vk::raii::CommandPool{device, graphicsCommandPoolInfo};
        vk::CommandPoolCreateInfo transferCommandPoolInfo {
            .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
            .queueFamilyIndex = transferIndex
        };
        transferCommandPool = vk::raii::CommandPool{device, transferCommandPoolInfo};
    }

    void createColorResources() {
        vk::Format colorFormat = swapChainImageFormat;

        createImage(
            swapChainExtent.width, swapChainExtent.height,
            1, msaaSamples, colorFormat,
            vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eTransientAttachment | vk::ImageUsageFlagBits::eColorAttachment,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            colorImage, colorImageMemory
        );
        colorImageView = createImageView(colorImage, colorFormat,
            vk::ImageAspectFlagBits::eColor, 1
        );
    }

    void createDepthResources() {
        vk::Format depthFormat = findDepthFormat();

        createImage(swapChainExtent.width, swapChainExtent.height,
            1, msaaSamples,
            depthFormat, vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eDepthStencilAttachment,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            depthImage, depthImageMemory
        );
        depthImageView = createImageView(depthImage, depthFormat, vk::ImageAspectFlagBits::eDepth, 1);
    }

    void createTextureImage() {
        int texWidth, texHeight, texChannels;
        // std::println("{}", (appDir / TEXTURE_PATH).string());
        stbi_uc* pixels = stbi_load((appDir / TEXTURE_PATH).c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
        if (!pixels) {
            throw std::runtime_error("Failed to load texture image!");
        }
        mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(texWidth, texHeight)))) + 1;

        vk::DeviceSize imageSize = texWidth * texHeight * 4;
        vk::raii::Buffer stagingBuffer {{}};
        vk::raii::DeviceMemory stagingBufferMemory {{}};
        createBuffer(imageSize, vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
            stagingBuffer, stagingBufferMemory
        );

        void* data = stagingBufferMemory.mapMemory(0, imageSize);
        memcpy(data, pixels, static_cast<size_t>(imageSize));
        stagingBufferMemory.unmapMemory();

        stbi_image_free(pixels);

        createImage(texWidth, texHeight,
            mipLevels, vk::SampleCountFlagBits::e1,
            vk::Format::eR8G8B8A8Srgb, vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            textureImage, textureImageMemory
        );

        transitionImageLayout(
            textureImage,
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::eTransferDstOptimal,
            mipLevels
        );
        copyBufferToImage(stagingBuffer, textureImage,
            static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight)
        );
        generateMipmaps(textureImage, vk::Format::eR8G8B8A8Srgb, texWidth, texHeight, mipLevels);
    }

    void createTextureImageView() {
        textureImageView = createImageView(textureImage, vk::Format::eR8G8B8A8Srgb, vk::ImageAspectFlagBits::eColor, mipLevels);
    }

    void createTextureSampler() {
        vk::PhysicalDeviceProperties properties = physicalDevice->getProperties();
        vk::SamplerCreateInfo samplerInfo {
            .magFilter = vk::Filter::eLinear,
            .minFilter = vk::Filter::eLinear,
            .mipmapMode = vk::SamplerMipmapMode::eLinear,
            .addressModeU = vk::SamplerAddressMode::eRepeat,
            .addressModeV = vk::SamplerAddressMode::eRepeat,
            .addressModeW = vk::SamplerAddressMode::eRepeat,
            .mipLodBias = 0.0f,
            .anisotropyEnable = vk::True,
            .maxAnisotropy = properties.limits.maxSamplerAnisotropy,
            .compareEnable = VK_FALSE,
            .compareOp = vk::CompareOp::eAlways,
            .minLod = 0.0f, // Optional
            .maxLod = static_cast<float>(mipLevels),
            .borderColor = vk::BorderColor::eIntOpaqueBlack,
            .unnormalizedCoordinates = vk::False,
        };
        textureSampler = vk::raii::Sampler {device, samplerInfo};
    }

    void loadModel(const std::string& modelName) {
        std::filesystem::path modelDir {appDir / "models" / modelName};
        std::filesystem::path objPath {};
        for (const auto& entry : std::filesystem::directory_iterator(modelDir)) {
            if (entry.is_regular_file() && entry.path().extension() == ".obj") {
                if (objPath.empty()) {
                    objPath = entry.path();
                } else {
                    throw std::runtime_error(std::format("Directory {} has more than one obj file.", modelName));
                }
            }
        }
        if (objPath.empty()) {
            throw std::runtime_error(std::format("Directory {} has no obj file.", modelName));
        }

        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;
        std::string warn, err;

        bool success = tinyobj::LoadObj(
            &attrib, &shapes, &materials,
            &warn, &err,
            objPath.c_str(), modelDir.c_str()
        );

        if (!success) {
            throw std::runtime_error(warn + err);
        }

        std::unordered_map<Vertex, uint32_t> uniqueVertices;
        for (const auto& shape : shapes) {
            for (const auto& index : shape.mesh.indices) {
                Vertex vertex {
                    .pos {
                        attrib.vertices[3 * index.vertex_index + 0],
                        attrib.vertices[3 * index.vertex_index + 1],
                        attrib.vertices[3 * index.vertex_index + 2]
                    },
                    .color {
                        1.0f, 1.0f, 1.0f
                    },
                    .texCoord {
                        attrib.texcoords[2 * index.texcoord_index + 0],
                        1.0f - attrib.texcoords[2 * index.texcoord_index + 1]
                    },
                };
                minX = std::min(minX, attrib.vertices[3 * index.vertex_index + 0]);
                maxX = std::max(maxX, attrib.vertices[3 * index.vertex_index + 0]);
                minY = std::min(minY, attrib.vertices[3 * index.vertex_index + 1]);
                maxY = std::max(maxY, attrib.vertices[3 * index.vertex_index + 1]);
                minZ = std::min(minZ, attrib.vertices[3 * index.vertex_index + 2]);
                maxZ = std::max(maxZ, attrib.vertices[3 * index.vertex_index + 2]);

                if (!uniqueVertices.contains(vertex)) {
                    uniqueVertices[vertex] = static_cast<uint32_t>(vertices.size());
                    vertices.emplace_back(vertex);
                }
                indices.emplace_back(uniqueVertices[vertex]);
            }
        }
    }

    void createVertexBuffer() {
        const vk::DeviceSize bufferSize = sizeof(Vertex) * vertices.size();

        vk::raii::Buffer stagingBuffer {nullptr};
        vk::raii::DeviceMemory stagingBufferMemory {nullptr};
        createBuffer(bufferSize,
            vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent,
            stagingBuffer, stagingBufferMemory
        );

        void* data = stagingBufferMemory.mapMemory(0, bufferSize);
        memcpy(data, vertices.data(), static_cast<size_t>(bufferSize));
        stagingBufferMemory.unmapMemory();

        createBuffer(bufferSize,
            vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            vertexBuffer, vertexBufferMemory
        );

        copyBuffer(stagingBuffer, vertexBuffer, bufferSize);
    }

    void createIndexBuffer() {
        const vk::DeviceSize bufferSize = sizeof(indices[0]) * indices.size();

        vk::raii::Buffer stagingBuffer {nullptr};
        vk::raii::DeviceMemory stagingBufferMemory {nullptr};
        createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
            stagingBuffer, stagingBufferMemory
        );

        void* data = stagingBufferMemory.mapMemory(0, bufferSize);
        memcpy(data, indices.data(), static_cast<size_t>(bufferSize));
        stagingBufferMemory.unmapMemory();

        createBuffer(bufferSize,
            vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            indexBuffer, indexBufferMemory
        );

        copyBuffer(stagingBuffer, indexBuffer, bufferSize);
    }

    void createUniformBuffers() {
        uniformBuffers.clear();
        uniformBuffersMemory.clear();
        uniformBuffersMapped.clear();

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
            constexpr vk::DeviceSize bufferSize = sizeof(UniformBufferObject);
            vk::raii::Buffer buffer {{}};
            vk::raii::DeviceMemory bufferMemory {{}};
            createBuffer(bufferSize, vk::BufferUsageFlagBits::eUniformBuffer,
                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                buffer, bufferMemory
            );
            uniformBuffers.emplace_back(std::move(buffer));
            uniformBuffersMemory.emplace_back(std::move(bufferMemory));
            uniformBuffersMapped.emplace_back(uniformBuffersMemory[i].mapMemory(0, bufferSize));
        }
    }

    void createDescriptorPool() {
        std::array poolSizes {
            vk::DescriptorPoolSize{
                .type = vk::DescriptorType::eUniformBuffer,
                .descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT)
            },
            vk::DescriptorPoolSize{
                .type = vk::DescriptorType::eCombinedImageSampler,
                .descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT)
            }
        };

        vk::DescriptorPoolCreateInfo poolInfo {
            .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
            .maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT),
            .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
            .pPoolSizes = poolSizes.data(),
        };

        descriptorPool = vk::raii::DescriptorPool{device, poolInfo};
    }

    void createDescriptorSets() {
        std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, descriptorSetLayout);
        vk::DescriptorSetAllocateInfo allocInfo {
            .descriptorPool = descriptorPool,
            .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
            .pSetLayouts = layouts.data()
        };

        descriptorSets.clear();
        descriptorSets = device.allocateDescriptorSets(allocInfo);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
            vk::DescriptorBufferInfo bufferInfo {
                .buffer = uniformBuffers[i],
                .offset = 0,
                .range = sizeof(UniformBufferObject)
            };

            vk::DescriptorImageInfo imageInfo {
                .sampler = textureSampler,
                .imageView = textureImageView,
                .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
            };

            std::array descriptorWrites {
                vk::WriteDescriptorSet{
                    .dstSet = descriptorSets[i],
                    .dstBinding = 0,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = vk::DescriptorType::eUniformBuffer,
                    .pBufferInfo = &bufferInfo,
                },
                vk::WriteDescriptorSet{
                    .dstSet = descriptorSets[i],
                    .dstBinding = 1,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                    .pImageInfo = &imageInfo
                }
            };

            device.updateDescriptorSets(descriptorWrites, {});
        }
    }

    void createCommandBuffers() {
        graphicsCommandBuffers.clear();
        vk::CommandBufferAllocateInfo allocInfo {
            .commandPool = graphicsCommandPool,
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT)
        };

        graphicsCommandBuffers = vk::raii::CommandBuffers(device, allocInfo);
    }

    void createSyncObjects() {
        presentCompleteSemaphores.clear();
        renderFinishedSemaphores.clear();
        inFlightFences.clear();

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
            presentCompleteSemaphores.emplace_back(device, vk::SemaphoreCreateInfo{});
            renderFinishedSemaphores.emplace_back(device, vk::SemaphoreCreateInfo{});
            inFlightFences.emplace_back(device, vk::FenceCreateInfo{.flags=vk::FenceCreateFlagBits::eSignaled});
        }
    }

    void recordCommandBuffer(uint32_t imageIndex) const {
        graphicsCommandBuffers[currentFrame].begin({});

        vk::ClearValue clearColor = vk::ClearColorValue{0.0f, 0.0f, 0.0f, 1.0f};
        vk::ClearValue clearDepth = vk::ClearDepthStencilValue(1.0f, 0);
        vk::ClearValue clearResolve = vk::ClearColorValue{0.0f, 0.0f, 0.0f, 1.0f};
        std::array clearValues = { clearColor, clearDepth, clearResolve};

        if (appInfo.dynamicRenderingSupported) {
            // begin dynamic rendering
            transitionImageLayout(
                swapChainImages[imageIndex],
                vk::ImageLayout::eUndefined,
                vk::ImageLayout::eColorAttachmentOptimal,
                {},
                vk::AccessFlagBits2::eColorAttachmentWrite,
                vk::PipelineStageFlagBits2::eTopOfPipe,
                vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                vk::ImageAspectFlagBits::eColor,
                1
            );
            // multisampled color image
            transitionImageLayout(
                colorImage,
                vk::ImageLayout::eUndefined,
                vk::ImageLayout::eColorAttachmentOptimal,
                {},
                vk::AccessFlagBits2::eColorAttachmentWrite,
                vk::PipelineStageFlagBits2::eTopOfPipe,
                vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                vk::ImageAspectFlagBits::eColor,
                1
            );
            // depth image
            transitionImageLayout(
                depthImage,
                vk::ImageLayout::eUndefined,
                vk::ImageLayout::eDepthAttachmentOptimal,
                {},
                vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
                vk::PipelineStageFlagBits2::eTopOfPipe,
                vk::PipelineStageFlagBits2::eEarlyFragmentTests,
                vk::ImageAspectFlagBits::eDepth,
                1
            );

            vk::RenderingAttachmentInfo colorAttachmentInfo {
                .imageView = colorImageView,
                .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                .resolveMode = vk::ResolveModeFlagBits::eAverage,
                .resolveImageView = swapChainImageViews[imageIndex],
                .resolveImageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                .loadOp = vk::AttachmentLoadOp::eClear,
                .storeOp = vk::AttachmentStoreOp::eStore,
                .clearValue = clearColor,
            };
            vk::RenderingAttachmentInfo depthAttachmentInfo {
                .imageView = depthImageView,
                .imageLayout = vk::ImageLayout::eDepthAttachmentOptimal,
                .loadOp = vk::AttachmentLoadOp::eClear,
                .storeOp = vk::AttachmentStoreOp::eDontCare,
                .clearValue = clearDepth,
            };

            vk::RenderingInfo renderingInfo {
                .renderArea = {
                    .offset = {0, 0}, .extent = swapChainExtent
                },
                .layerCount = 1,
                .colorAttachmentCount = 1,
                .pColorAttachments = &colorAttachmentInfo,
                .pDepthAttachment = &depthAttachmentInfo,
            };

            graphicsCommandBuffers[currentFrame].beginRendering(renderingInfo);
        } else {
            // Use traditional render pass
            vk::RenderPassBeginInfo renderPassBeginInfo {
                .renderPass = *renderPass,
                .framebuffer = *swapChainFramebuffers[imageIndex],
                .renderArea = {{0, 0}, swapChainExtent},
                .clearValueCount = static_cast<uint32_t>(clearValues.size()),
                .pClearValues = clearValues.data(),
            };

            graphicsCommandBuffers[currentFrame].beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);
        }

        graphicsCommandBuffers[currentFrame].setViewport(0,
            vk::Viewport{
                0.0f, 0.0f,
                static_cast<float>(swapChainExtent.width),
                static_cast<float>(swapChainExtent.height),
                0.0f, 1.0f
            });
        graphicsCommandBuffers[currentFrame].setScissor(0, vk::Rect2D{vk::Offset2D{0, 0}, swapChainExtent});
        graphicsCommandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eGraphics, *graphicsPipeline);
        graphicsCommandBuffers[currentFrame].bindVertexBuffers(0, {*vertexBuffer}, {0});
        graphicsCommandBuffers[currentFrame].bindIndexBuffer(*indexBuffer, 0, vk::IndexType::eUint32);
        graphicsCommandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
            *pipelineLayout, 0, {*descriptorSets[currentFrame]}, {});
        graphicsCommandBuffers[currentFrame].drawIndexed(static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);

        if (appInfo.dynamicRenderingSupported) {
            graphicsCommandBuffers[currentFrame].endRendering();

            transitionImageLayout(
                swapChainImages[imageIndex],
                vk::ImageLayout::eColorAttachmentOptimal,
                vk::ImageLayout::ePresentSrcKHR,
                1
            );
        } else {
            graphicsCommandBuffers[currentFrame].endRenderPass();
        }

        graphicsCommandBuffers[currentFrame].end();
    }

    void updateUniformBuffer(uint32_t currentImage) const {
        static auto startTime = std::chrono::high_resolution_clock::now();

        const auto currentTime = std::chrono::high_resolution_clock::now();
        const float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

        glm::vec3 objCenter {(minX + maxX) / 2.0f, (minY + maxY) / 2.0f, (minZ + maxZ) / 2.0f };
        float xLen { maxX - minX }, yLen { maxY - minY }, zLen { maxZ - minZ };
        glm::vec3 eye { xLen * 1.5, yLen * 1.5, zLen * 1.5};
        glm::vec3 up {0, 1, 0};

        UniformBufferObject ubo {
            .model = glm::rotate(
                glm::translate(glm::mat4(1.0f), -objCenter),
                time * glm::radians(45.0f),
                glm::vec3(0.0f, 1.0f, 0.0f)
            ),
            .view = glm::lookAt(eye, glm::vec3(0.0f), up),
            .proj = glm::perspective(glm::radians(45.0f),
                static_cast<float>(swapChainExtent.width) / static_cast<float>(swapChainExtent.height),
                0.01f, static_cast<float>(glm::length(glm::vec3{xLen, yLen, zLen})) * 3
            )
        };
        // GLM was designed for OpenGL
        // where Y coordinate of the clip coordinates is inverted
        ubo.proj[1][1] *= -1;

        memcpy(uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
    }

    void drawFrame() {
        while (vk::Result::eTimeout == device.waitForFences(*inFlightFences[currentFrame], vk::True, UINT64_MAX))
            ;
        // Acquire an image from the swap chain
        auto [result, imageIndex] = swapChain.acquireNextImage(
            UINT64_MAX, presentCompleteSemaphores[currentFrame], nullptr);
        if (result == vk::Result::eErrorOutOfDateKHR) {
            framebufferResized = false;
            recreateSwapChain();
            return;
        } else if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR) {
            throw std::runtime_error("Failed to acquire swap chain image!");
        }

        updateUniformBuffer(currentFrame);
        // Only reset the fence if we are submitting work
        device.resetFences(*inFlightFences[currentFrame]);

        // Record the command buffer
        graphicsCommandBuffers[currentFrame].reset();
        recordCommandBuffer(imageIndex);

        // Submit the command buffer
        // Traditional binary semaphores
        vk::PipelineStageFlags waitDestinationStageMask {
            vk::PipelineStageFlagBits::eColorAttachmentOutput
        };
        const vk::SubmitInfo submitInfo {
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &*presentCompleteSemaphores[currentFrame],
            .pWaitDstStageMask = &waitDestinationStageMask,
            .commandBufferCount = 1,
            .pCommandBuffers = &*graphicsCommandBuffers[currentFrame],
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &*renderFinishedSemaphores[currentFrame],
        };

        graphicsQueue.submit(submitInfo, *inFlightFences[currentFrame]);

        // Presentation
        const vk::PresentInfoKHR presentInfo {
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &*renderFinishedSemaphores[currentFrame],
            .swapchainCount = 1,
            .pSwapchains = &*swapChain,
            .pImageIndices = &imageIndex,
            .pResults = nullptr, // To check multiple swap chains' results
        };
        result = presentQueue.presentKHR(presentInfo);
        if (result == vk::Result::eErrorOutOfDateKHR ||
            result == vk::Result::eSuboptimalKHR ||
            framebufferResized) {
            framebufferResized = false;
            recreateSwapChain();
        } else if (result != vk::Result::eSuccess) {
            throw std::runtime_error("Failed to present swap chain image!");
        }

        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    void recreateSwapChain() {
        int width = 0, height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        while (width == 0 || height == 0) {
            glfwGetFramebufferSize(window, &width, &height);
            glfwWaitEvents();
        };

        device.waitIdle();

        cleanupSwapChain();

        createSwapChain();
        createImageViews();
        createColorResources();
        createDepthResources();
        // createFramebuffers();
    }


    /**
     * Helper functions
     */

    // proxy
    // vk::Result CreateDebugUtilsMessengerEXT(VkInstance instance,
    //     const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
    //     const VkAllocationCallbacks* pAllocator,
    //     VkDebugUtilsMessengerEXT* pDebugMessenger) {
    //     auto func = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>
    //         (vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
    //     if (func != nullptr) {
    //         return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    //     } else {
    //         return vk::Result::eErrorExtensionNotPresent;
    //     }
    // }

    // proxy
    // void DestroyDebugUtilsMessengerEXT(VkInstance instance,
    //     VkDebugUtilsMessengerEXT debugMessenger,
    //     const VkAllocationCallbacks* pAllocator) {
    //     auto func = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>
    //         (vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
    //     if (func != nullptr) {
    //         func(instance, debugMessenger, pAllocator);
    //     }
    // }
    //
    // void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT &createInfo) {
    //     createInfo = {
    //         .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
    //         .messageSeverity =
    //             VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
    //             VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
    //             VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
    //         .messageType =
    //             VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
    //             VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
    //             VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
    //         .pfnUserCallback = debugCallback,
    //         .pUserData = nullptr
    //     };
    // }

    static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
        // std::println("resize callback");
        auto app = static_cast<Application*>(glfwGetWindowUserPointer(window));
        app->framebufferResized = true;
    }

    bool isDeviceSuitable(const vk::raii::PhysicalDevice& device) {
        auto availableExtensions {device.enumerateDeviceExtensionProperties()};
        auto deviceProperties {device.getProperties()};
        auto deviceFeatures {device.getFeatures()};
        auto queueFamilies {device.getQueueFamilyProperties()};

        bool extensionSupported = std::ranges::all_of(requiredDeviceExtensions,
            [&availableExtensions](const char* deviceExtension) {
            return std::ranges::any_of(availableExtensions, [&deviceExtension](const auto& availableExtension) {
                return strcmp(deviceExtension, availableExtension.extensionName) == 0;
            });
        });
        bool queueFamilySupported = std::ranges::any_of(queueFamilies,
            [](const vk::QueueFamilyProperties& qfp) {
                return (qfp.queueFlags & vk::QueueFlagBits::eGraphics) != static_cast<vk::QueueFlagBits>(0);
            }
        );
        bool swapChainAdequate = false;
        if (extensionSupported) {
            auto formats {device.getSurfaceFormatsKHR(*surface)};
            auto presentModes {device.getSurfacePresentModesKHR(*surface)};
            // at least one format and one present mode must be available
            swapChainAdequate = !formats.empty() && !presentModes.empty();
        }


        return
            deviceProperties.apiVersion >= vk::ApiVersion13 &&
            queueFamilySupported &&
            extensionSupported &&
            swapChainAdequate &&
            // deviceFeatures.geometryShader &&
            deviceFeatures.samplerAnisotropy;
    }

    [[nodiscard]] std::vector<const char*> getRequiredLayers() const {
        std::vector<const char*> requiredLayers {};
        checkLayerSupport(requiredLayers);
        return requiredLayers;
    }

    void checkLayerSupport(const std::vector<const char*>& requiredLayers) const {
        const auto layerProperties = context.enumerateInstanceLayerProperties();
        if (std::ranges::any_of(requiredLayers, [&layerProperties](const auto& requiredLayer) {
            return std::ranges::none_of(layerProperties, [&requiredLayer](const auto& layerProperty) {
                return strcmp(requiredLayer, layerProperty.layerName) == 0;
            });
        })) {
            throw std::runtime_error("One or more required layer are not supported!");
        }
    }

    [[nodiscard]] std::vector<const char*> getRequiredExtensions() const {
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
        std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

        // Check if debug utils extension is available
        auto props = context.enumerateInstanceExtensionProperties();
        bool debugUtilsAvailable = std::ranges::any_of(props,
            [](const auto& prop) {
                return strcmp(prop.extensionName, vk::EXTDebugUtilsExtensionName) == 0;
            });
        // Always include the debug utils extension if available
        // This allows the validation layer via vkconfig
        if (debugUtilsAvailable) {
            extensions.emplace_back(vk::EXTDebugUtilsExtensionName);
        } else {
            std::println("VK_EXT_debug_utils extension not available. Validation layers may not work.");
        }

#ifdef __APPLE__
        extensions.emplace_back(vk::KHRPortabilityEnumerationExtensionName);
        // needed by VK_KHR_portability_subset
        extensions.emplace_back(vk::KHRGetPhysicalDeviceProperties2ExtensionName);
#endif

        checkExtensionSupport(extensions);
        return extensions;
    }

    void checkExtensionSupport(const std::vector<const char*>& requiredExtensions) const {
        auto extensionProperties = context.enumerateInstanceExtensionProperties();

        // check
        for (const auto& requiredExtension : requiredExtensions) {
            if (std::ranges::none_of(extensionProperties,
                [requiredExtension](const auto& extension) {
                    return strcmp(requiredExtension, extension.extensionName) == 0;
            })) {
                throw std::runtime_error("Required extension not support: " + std::string(requiredExtension));
            }
        }
    }

    uint32_t rateDeviceSuitability(const vk::raii::PhysicalDevice& device) {
        if (!isDeviceSuitable(device)) {
            return 0;
        }

        vk::PhysicalDeviceProperties deviceProperties {device.getProperties()};

        uint32_t score = 0;
        // discrete GPUs have significant performance advantage
        if (deviceProperties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu) {
            score += 1000;
        }

        score += deviceProperties.limits.maxImageDimension2D;

        return score;
    }

    [[nodiscard]] QueueFamilyIndices findQueueFamilies(const vk::raii::PhysicalDevice& physicalDevice) const {
        QueueFamilyIndices indices;
        const auto queueFamilyProperties = physicalDevice.getQueueFamilyProperties();

        int i = 0;
        for (const auto& qfp : queueFamilyProperties) {
            bool graphicsSupport = (qfp.queueFlags & vk::QueueFlagBits::eGraphics) != static_cast<vk::QueueFlagBits>(0);
            bool transferSupport = (qfp.queueFlags & vk::QueueFlagBits::eTransfer) != static_cast<vk::QueueFlagBits>(0);
            VkBool32 presentSupport = physicalDevice.getSurfaceSupportKHR(i, surface);

            if (!indices.graphicsFamily.has_value() && graphicsSupport) {
                indices.graphicsFamily = i;
            }
            if (!indices.presentFamily.has_value() && presentSupport) {
                indices.presentFamily = i;
            }
            // prefer transfer only
            if (!indices.transferFamily.has_value() && !graphicsSupport && transferSupport) {
                indices.transferFamily = i;
            }

            if (indices.isComplete()) {
                break;
            }
            ++i;
        }

        i = 0;
        if (!indices.transferFamily.has_value()) {
            for (const auto& qfp : queueFamilyProperties) {
                bool transferSupport = (qfp.queueFlags & vk::QueueFlagBits::eTransfer) == static_cast<vk::QueueFlagBits>(0);
                if (indices.graphicsFamily.value() != i && transferSupport) {
                    indices.transferFamily = i;
                    break;
                }
                ++i;
            }
        }

        if (!indices.transferFamily.has_value()) {
            indices.transferFamily = indices.graphicsFamily.value();
        }

        if (!indices.isComplete()) {
            throw std::runtime_error("Failed to find all queue families.");
        }

        return indices;
    }

    [[nodiscard]] uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) const {
        vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice->getMemoryProperties();

        for (uint32_t i = 0; i < memProperties.memoryTypeCount; ++i) {
            if (typeFilter & (1 << i) &&
                (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
                }
        }

        throw std::runtime_error("Failed to find suitable memory type!");
    }

    [[nodiscard]] vk::Format findSupportedFormat(
        const std::vector<vk::Format>& candidates,
        vk::ImageTiling tiling,
        vk::FormatFeatureFlags features
    ) const {
        for (const auto format : candidates) {
            vk::FormatProperties props = physicalDevice->getFormatProperties(format);
            if (tiling == vk::ImageTiling::eLinear &&
                (props.linearTilingFeatures & features) == features) {
                    return format;
                } else if (tiling == vk::ImageTiling::eOptimal && props.optimalTilingFeatures & features) {
                    return format;
                }
        }

        throw std::runtime_error("Failed to find supported format!");
    }

    [[nodiscard]] vk::Format findDepthFormat() const {
        return findSupportedFormat(
            { vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint},
            vk::ImageTiling::eOptimal,
           vk::FormatFeatureFlagBits::eDepthStencilAttachment
        );
    }

    [[nodiscard]] vk::SampleCountFlagBits getMaxUsableSampleCount() const {
        vk::PhysicalDeviceProperties physicalDeviceProperties = physicalDevice->getProperties();

        vk::SampleCountFlags counts {
            physicalDeviceProperties.limits.framebufferColorSampleCounts &
            physicalDeviceProperties.limits.framebufferDepthSampleCounts
        };

        if (counts & vk::SampleCountFlagBits::e64) {
            std::println("Sample count: 64");
            return vk::SampleCountFlagBits::e64;
        } else if (counts & vk::SampleCountFlagBits::e32) {
            std::println("Sample count: 32");
            return vk::SampleCountFlagBits::e32;
        } else if (counts & vk::SampleCountFlagBits::e16) {
            std::println("Sample count: 16");
            return vk::SampleCountFlagBits::e16;
        } else if (counts & vk::SampleCountFlagBits::e8) {
            std::println("Sample count: 8");
            return vk::SampleCountFlagBits::e8;
        } else if (counts & vk::SampleCountFlagBits::e4) {
            std::println("Sample count: 4");
            return vk::SampleCountFlagBits::e4;
        } else if (counts & vk::SampleCountFlagBits::e2) {
            std::println("Sample count: 2");
            return vk::SampleCountFlagBits::e2;
        } else {
            std::println("Sample count: 1");
            return vk::SampleCountFlagBits::e1;
        }
    }

    [[nodiscard]] vk::Format chooseSwapSurfaceFormat(
        const std::vector<vk::SurfaceFormatKHR>& availableFormats) const noexcept{
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

    [[nodiscard]] vk::PresentModeKHR chooseSwapPresentMode(
        const std::vector<vk::PresentModeKHR>& availablePresentModes
    ) const noexcept{
        for (const auto& availablePresentMode : availablePresentModes) {
            if (availablePresentMode == vk::PresentModeKHR::eMailbox) {
                // relatively high energy consumption, but low latency
                return availablePresentMode; // prefer mailbox mode for lower latency
            }
        }

        return vk::PresentModeKHR::eFifo;
    }

    [[nodiscard]] vk::Extent2D chooseSwapExtent(
        const vk::SurfaceCapabilitiesKHR& capabilities) const noexcept{
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
            return capabilities.currentExtent;
        } else {
            int width, height;
            glfwGetFramebufferSize(window, &width, &height);

            return {
                .width = std::clamp<uint32_t>(width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
                .height = std::clamp<uint32_t>(height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height),
            };
        }
    }

    [[nodiscard]] vk::raii::ShaderModule createShaderModule(const std::vector<char>& code) const{
        vk::ShaderModuleCreateInfo createInfo {
            .codeSize = code.size() * sizeof(char),
            .pCode = reinterpret_cast<const uint32_t*>(code.data())
        };

        return vk::raii::ShaderModule{device, createInfo};
    }


    void createBuffer(
        vk::DeviceSize size,
        vk::BufferUsageFlags usage,
        vk::MemoryPropertyFlags properties,
        vk::raii::Buffer& buffer,
        vk::raii::DeviceMemory& bufferMemory
    ) const {
        uint32_t queueFamilyIndices[] = {
            graphicsIndex, transferIndex
        };

        vk::BufferCreateInfo bufferInfo {
            .size = size,
            .usage = usage
        };
        if (graphicsIndex == transferIndex) {
            bufferInfo.sharingMode = vk::SharingMode::eExclusive;
            bufferInfo.queueFamilyIndexCount = 0;
            bufferInfo.pQueueFamilyIndices = nullptr;
        } else {
            bufferInfo.sharingMode = vk::SharingMode::eConcurrent;
            bufferInfo.queueFamilyIndexCount = 2;
            bufferInfo.pQueueFamilyIndices = queueFamilyIndices;
        }

        buffer = vk::raii::Buffer{device, bufferInfo};

        vk::MemoryRequirements memRequirements = buffer.getMemoryRequirements();

        vk::MemoryAllocateInfo allocInfo {
            .allocationSize = memRequirements.size,
            .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties)
        };
        bufferMemory = vk::raii::DeviceMemory{device, allocInfo};
        buffer.bindMemory(bufferMemory, 0);
    }

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
    ) const {
        vk::ImageCreateInfo imageInfo {
            .imageType = vk::ImageType::e2D,
            .format = format,
            .extent = {
                .width = width,
                .height = height,
                .depth = 1
            },
            .mipLevels = mipLevels,
            .arrayLayers = 1,
            .samples = numSamples,
            .tiling = tiling,
            .usage = usage,
            .sharingMode = vk::SharingMode::eExclusive,
            .initialLayout = vk::ImageLayout::eUndefined,
        };

        image = vk::raii::Image{device, imageInfo};

        vk::MemoryRequirements memRequirements = image.getMemoryRequirements();

        vk::MemoryAllocateInfo allocInfo {
            .allocationSize = memRequirements.size,
            .memoryTypeIndex = findMemoryType(
                memRequirements.memoryTypeBits, properties
            )
        };
        imageMemory = vk::raii::DeviceMemory{device, allocInfo};
        image.bindMemory(*imageMemory, 0);
    }

    [[nodiscard]] vk::raii::ImageView createImageView(
        const vk::raii::Image& image,
        vk::Format format,
        vk::ImageAspectFlags aspectFlags,
        uint32_t mipLevels
    ) const {
        vk::ImageViewCreateInfo viewInfo {
            .image = image,
            .viewType = vk::ImageViewType::e2D,
            .format = format,
            .subresourceRange = {
                .aspectMask = aspectFlags,
                .baseMipLevel = 0,
                .levelCount = mipLevels,
                .baseArrayLayer = 0,
                .layerCount = 1
            }
        };

        return {device, viewInfo};
    }

    [[nodiscard]] vk::raii::CommandBuffer beginSingleTimeCommands(const vk::raii::CommandPool& commandPool) const {
        vk::CommandBufferAllocateInfo allocInfo {
            .commandPool = commandPool,
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = 1
        };

        vk::raii::CommandBuffer commandBuffer = std::move(device.allocateCommandBuffers(allocInfo).front());
        commandBuffer.begin(vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

        return commandBuffer;
    }

    void endSingleTimeCommands(
        const vk::raii::CommandBuffer& commandBuffer,
        const vk::raii::Queue& queue
    ) const {
        commandBuffer.end();

        queue.submit(vk::SubmitInfo{.commandBufferCount = 1, .pCommandBuffers = &*commandBuffer}, nullptr);
        queue.waitIdle();
    }

    void copyBuffer(
        const vk::raii::Buffer& srcBuffer,
        const vk::raii::Buffer& dstBuffer,
        vk::DeviceSize size
    ) const {
        vk::raii::CommandBuffer commandCopyBuffer = beginSingleTimeCommands(transferCommandPool);
        commandCopyBuffer.copyBuffer(srcBuffer, dstBuffer, vk::BufferCopy{0, 0, size});
        endSingleTimeCommands(commandCopyBuffer, transferQueue);
    }

    void transitionImageLayout(
        const vk::raii::Image& image,
        // vk::Format format,
        vk::ImageLayout oldLayout,
        vk::ImageLayout newLayout,
        uint32_t mipLevels
    ) const {
        vk::ImageAspectFlagBits aspectMask;
        if (newLayout == vk::ImageLayout::eDepthStencilAttachmentOptimal) {
            aspectMask = vk::ImageAspectFlagBits::eDepth;
            // if (vk::hasStencilComponent(format)) {
            //     barrier.subresourceRange.aspectMask |= vk::ImageAspectFlagBits::eStencil;
            // }
        } else {
            aspectMask = vk::ImageAspectFlagBits::eColor;
        }

        if (oldLayout == vk::ImageLayout::eUndefined &&
            newLayout == vk::ImageLayout::eTransferDstOptimal
        ) {
            transitionImageLayout(image, oldLayout, newLayout,
                {},
                vk::AccessFlagBits2::eTransferWrite,
                vk::PipelineStageFlagBits2::eTopOfPipe,
                vk::PipelineStageFlagBits2::eTransfer,
                aspectMask, mipLevels);
        } else if (oldLayout == vk::ImageLayout::eTransferDstOptimal &&
            newLayout == vk::ImageLayout::eShaderReadOnlyOptimal
        ) {
            transitionImageLayout(image, oldLayout, newLayout,
                vk::AccessFlagBits2::eTransferWrite,
                vk::AccessFlagBits2::eShaderRead,
                vk::PipelineStageFlagBits2::eTransfer,
                vk::PipelineStageFlagBits2::eFragmentShader,
                aspectMask, mipLevels);
        } else if (oldLayout == vk::ImageLayout::eUndefined &&
            newLayout == vk::ImageLayout::eDepthStencilAttachmentOptimal
        ) {
            transitionImageLayout(image, oldLayout, newLayout,
                {},
                vk::AccessFlagBits2::eDepthStencilAttachmentRead | vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
                vk::PipelineStageFlagBits2::eTopOfPipe,
                vk::PipelineStageFlagBits2::eEarlyFragmentTests,
                aspectMask, mipLevels);
        } else if (oldLayout == vk::ImageLayout::eUndefined &&
            newLayout == vk::ImageLayout::eColorAttachmentOptimal
        ) {
            transitionImageLayout(image, oldLayout, newLayout,
                {},
                vk::AccessFlagBits2::eColorAttachmentWrite,
                vk::PipelineStageFlagBits2::eTopOfPipe,
                vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                aspectMask, mipLevels);
        } else if (oldLayout == vk::ImageLayout::eColorAttachmentOptimal &&
            newLayout == vk::ImageLayout::ePresentSrcKHR
        ) {
            transitionImageLayout(image, oldLayout, newLayout,
                vk::AccessFlagBits2::eColorAttachmentWrite,
                {},
                vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                vk::PipelineStageFlagBits2::eBottomOfPipe,
                aspectMask, mipLevels);
        } else {
            throw std::invalid_argument("unsupported layout transition!");
        }
    }

    void transitionImageLayout(
        const vk::Image& image,
        // vk::Format format,
        vk::ImageLayout oldLayout,
        vk::ImageLayout newLayout,
        uint32_t mipLevels
    ) const {
        vk::ImageAspectFlagBits aspectMask;
        if (newLayout == vk::ImageLayout::eDepthStencilAttachmentOptimal) {
            aspectMask = vk::ImageAspectFlagBits::eDepth;
            // if (vk::hasStencilComponent(format)) {
            //     barrier.subresourceRange.aspectMask |= vk::ImageAspectFlagBits::eStencil;
            // }
        } else {
            aspectMask = vk::ImageAspectFlagBits::eColor;
        }

        if (oldLayout == vk::ImageLayout::eUndefined &&
            newLayout == vk::ImageLayout::eTransferDstOptimal
        ) {
            transitionImageLayout(image, oldLayout, newLayout,
                {},
                vk::AccessFlagBits2::eTransferWrite,
                vk::PipelineStageFlagBits2::eTopOfPipe,
                vk::PipelineStageFlagBits2::eTransfer,
                aspectMask, mipLevels);
        } else if (oldLayout == vk::ImageLayout::eTransferDstOptimal &&
            newLayout == vk::ImageLayout::eShaderReadOnlyOptimal
        ) {
            transitionImageLayout(image, oldLayout, newLayout,
                vk::AccessFlagBits2::eTransferWrite,
                vk::AccessFlagBits2::eShaderRead,
                vk::PipelineStageFlagBits2::eTransfer,
                vk::PipelineStageFlagBits2::eFragmentShader,
                aspectMask, mipLevels);
        } else if (oldLayout == vk::ImageLayout::eUndefined &&
            newLayout == vk::ImageLayout::eDepthStencilAttachmentOptimal
        ) {
            transitionImageLayout(image, oldLayout, newLayout,
                {},
                vk::AccessFlagBits2::eDepthStencilAttachmentRead | vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
                vk::PipelineStageFlagBits2::eTopOfPipe,
                vk::PipelineStageFlagBits2::eEarlyFragmentTests,
                aspectMask, mipLevels);
        } else if (oldLayout == vk::ImageLayout::eUndefined &&
            newLayout == vk::ImageLayout::eColorAttachmentOptimal
        ) {
            transitionImageLayout(image, oldLayout, newLayout,
                {},
                vk::AccessFlagBits2::eColorAttachmentWrite,
                vk::PipelineStageFlagBits2::eTopOfPipe,
                vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                aspectMask, mipLevels);
        } else if (oldLayout == vk::ImageLayout::eColorAttachmentOptimal &&
            newLayout == vk::ImageLayout::ePresentSrcKHR
        ) {
            transitionImageLayout(image, oldLayout, newLayout,
                vk::AccessFlagBits2::eColorAttachmentWrite,
                {},
                vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                vk::PipelineStageFlagBits2::eBottomOfPipe,
                aspectMask, mipLevels);
        } else {
            throw std::invalid_argument("unsupported layout transition!");
        }
    }

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
    ) const {
        auto commandBuffer = beginSingleTimeCommands(transferCommandPool);
        vk::ImageMemoryBarrier2 barrier {
            .srcStageMask = srcStageMask,
            .srcAccessMask = srcAccessMask,
            .dstStageMask = dstStageMask,
            .dstAccessMask = dstAccessMask,
            .oldLayout = oldLayout,
            .newLayout = newLayout,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = *image,
            .subresourceRange = {
                .aspectMask = aspectMask,
                .baseMipLevel = 0,
                .levelCount = mipLevels,
                .baseArrayLayer = 0,
                .layerCount = 1
            },
        };
        vk::DependencyInfo dependencyInfo {
            .dependencyFlags = {},
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers = &barrier,
        };
        commandBuffer.pipelineBarrier2(dependencyInfo);
        endSingleTimeCommands(commandBuffer, transferQueue);
    }

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
    ) const {
        auto commandBuffer = beginSingleTimeCommands(transferCommandPool);
        vk::ImageMemoryBarrier2 barrier {
            .srcStageMask = srcStageMask,
            .srcAccessMask = srcAccessMask,
            .dstStageMask = dstStageMask,
            .dstAccessMask = dstAccessMask,
            .oldLayout = oldLayout,
            .newLayout = newLayout,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = image,
            .subresourceRange = {
                .aspectMask = aspectMask,
                .baseMipLevel = 0,
                .levelCount = mipLevels,
                .baseArrayLayer = 0,
                .layerCount = 1
            },
        };
        vk::DependencyInfo dependencyInfo {
            .dependencyFlags = {},
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers = &barrier,
        };
        commandBuffer.pipelineBarrier2(dependencyInfo);
        endSingleTimeCommands(commandBuffer, transferQueue);
    }

    void copyBufferToImage(
        const vk::raii::Buffer& buffer,
        const vk::raii::Image& image,
        uint32_t width,
        uint32_t height
    ) const {
        auto commandBuffer = beginSingleTimeCommands(transferCommandPool);

        vk::BufferImageCopy region {
            .bufferOffset = 0,
            .bufferRowLength = 0,
            .bufferImageHeight = 0,
            .imageSubresource = {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = 1
            },
            .imageOffset = {0, 0, 0},
            .imageExtent = {
                .width = width,
                .height = height,
                .depth = 1
            }
        };
        commandBuffer.copyBufferToImage(buffer, image, vk::ImageLayout::eTransferDstOptimal, {region});

        endSingleTimeCommands(commandBuffer, transferQueue);
    }

    void generateMipmaps(
        const vk::raii::Image& image,
        vk::Format imageFormat,
        int32_t texWidth,
        int32_t texHeight,
        uint32_t mipLevels
    ) const {
        // Check if the image format supports linear blitting
        vk::FormatProperties formatProperties = physicalDevice->getFormatProperties(imageFormat);
        if (!(formatProperties.optimalTilingFeatures & vk::FormatFeatureFlagBits::eSampledImageFilterLinear)) {
            throw std::runtime_error("Texture image format does not support linear blitting!");
        }
        // alternatives:
        // 1.   search common texture image formats for one
        //      that does support linear bitting
        // 2.   implement the mipmap generation in software with a library like stb_image_resize

        auto commandBuffer = beginSingleTimeCommands(graphicsCommandPool);

        vk::ImageMemoryBarrier barrier {
            .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
            .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
            .image = image,
            .subresourceRange = {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            }
        };

        int32_t mipWidth {texWidth}, mipHeight {texHeight};
        for (uint32_t i = 1; i < mipLevels; ++i) {
            barrier.subresourceRange.baseMipLevel = i - 1;
            barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
            barrier.newLayout = vk::ImageLayout::eTransferSrcOptimal;
            barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
            barrier.dstAccessMask = vk::AccessFlagBits::eTransferRead;

            commandBuffer.pipelineBarrier(
                vk::PipelineStageFlagBits::eTransfer,
                vk::PipelineStageFlagBits::eTransfer,
                {}, {}, {}, barrier);

            vk::ArrayWrapper1D<vk::Offset3D, 2> srcOffsets, dstOffsets;
            srcOffsets[0] = {0, 0, 0};
            srcOffsets[1] = {mipWidth, mipHeight, 1};
            dstOffsets[0] = {0, 0, 0};
            dstOffsets[1] = {mipWidth > 1 ? mipWidth / 2 : 1, mipHeight > 1 ? mipHeight / 2 : 1, 1};
            vk::ImageBlit blit {
                .srcSubresource = {vk::ImageAspectFlagBits::eColor, i -1 , 0, 1},
                .srcOffsets = srcOffsets,
                .dstSubresource = {vk::ImageAspectFlagBits::eColor, i, 0, 1},
                .dstOffsets = dstOffsets,
            };
            commandBuffer.blitImage(
                image, vk::ImageLayout::eTransferSrcOptimal,
                image, vk::ImageLayout::eTransferDstOptimal,
                {blit}, vk::Filter::eLinear);

            barrier.oldLayout = vk::ImageLayout::eTransferSrcOptimal;
            barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
            barrier.srcAccessMask = vk::AccessFlagBits::eTransferRead;
            barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

            commandBuffer.pipelineBarrier(
                vk::PipelineStageFlagBits::eTransfer,
                vk::PipelineStageFlagBits::eFragmentShader,
                {}, {}, {}, barrier);

            if (mipWidth > 1) mipWidth /= 2;
            if (mipHeight > 1) mipHeight /= 2;
        }

        barrier.subresourceRange.baseMipLevel = mipLevels - 1;
        barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
        barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
        barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

        commandBuffer.pipelineBarrier(
            vk::PipelineStageFlagBits::eTransfer,
            vk::PipelineStageFlagBits::eFragmentShader,
            {}, {}, {}, barrier);

        endSingleTimeCommands(commandBuffer, graphicsQueue);
    }
};

int main(int argc, char* argv[]) {
    const std::filesystem::path binPath(argv[0]);
    std::filesystem::path appDir = binPath.parent_path().parent_path();

    Application app(std::move(appDir));

    try {
        app.run();
    } catch (vk::SystemError& err) {
        std::println(stderr, "Vulkan error: {}", err.what());
    } catch (std::exception& err) {
        std::println(stderr, "{}", err.what());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}