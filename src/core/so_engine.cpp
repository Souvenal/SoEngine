#include "so_engine.h"

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include <tiny_gltf.h>

#define VMA_IMPLEMENTATION
#include <vma/vk_mem_alloc.h>

#include <set>
#include <iostream>
#include <fstream>
#include <algorithm>

static VKAPI_ATTR vk::Bool32 VKAPI_CALL debugCallback(
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

SoEngine::SoEngine(const std::filesystem::path& path): appDir(path) {}

void SoEngine::prepare(const Window* window) {
    createInstance();
    setupDebugMessenger();
    surface = window->createSurface(*instance);        // influence physical device selection
    auto framebufferSize = window->getFramebufferSize();
    swapChainExtent.width = framebufferSize.width;
    swapChainExtent.height = framebufferSize.height;
    pickPhysicalDevice();
    checkFeatureSupport();
    // detectFeatureSupport();
    createLogicalDevice();
    createMemoryAllocator();
    createSwapChain();
    std::println("Number of images in the swap chain: {}", swapChainImages.size());
    createImageViews();

    if (!appInfo.profileSupported && !appInfo.dynamicRenderingSupported) {
        createRenderPass();
    }
    createDescriptorSetLayout();
    createGraphicsPipeline();
    createCommandPool();

    createColorResources();
    createDepthResources();
    if (!appInfo.profileSupported && !appInfo.dynamicRenderingSupported) {
        createFramebuffers();
    }
    createTextureImage();
    createTextureImageView();
    createTextureSampler();

    // loadModel("amiya-arknights");
    loadModel("viking_room");
    setupModelObjects();
    createVertexBuffer();
    createIndexBuffer();

    createUniformBuffers();
    createDescriptorPool();
    createDescriptorSets();

    createFrameData();
    // Print feature support summary
    // appInfo.printFeatureSupportSummary();
}

void SoEngine::update(float deltaTime) {
    // while(!glfwWindowShouldClose(window)) {
    //     glfwPollEvents();
    //     drawFrame();
    // }

    // device.waitIdle();
    drawFrame();
}

bool SoEngine::shouldTerminate() const {
    return false;
}

void SoEngine::terminate() {
    device.waitIdle();

    cleanup();
}

void SoEngine::inputEvent(const InputEvent& event) {
    if (event.getSource() == EventSource::Mouse) {
        const auto& mouseButton = static_cast<const MouseButtonInputEvent&>(event);
    } else if (event.getSource() == EventSource::Keyboard) {
        const auto& keyEvent = static_cast<const KeyInputEvent&>(event);
    }
}

FrameData& SoEngine::getCurrentFrame() {
    return frames[currentFrame];
}

void SoEngine::cleanup() {
    cleanupSwapChain();
    mainDeletionQueue.flush();
}

void SoEngine::cleanupSwapChain() {
    swapChainImageViews.clear();
    swapChain = nullptr;
    resourceDeletionQueue.flush();
}

void SoEngine::createInstance() {
    constexpr vk::ApplicationInfo appInfo {
        .pApplicationName = "Vulkan application",
        .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
        .pEngineName = "So Engine",
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

void SoEngine::setupDebugMessenger() {
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

void SoEngine::pickPhysicalDevice() {
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
}

void SoEngine::checkFeatureSupport() {
    appInfo.profile = {
        VP_KHR_ROADMAP_2022_NAME,
        VP_KHR_ROADMAP_2022_SPEC_VERSION
    };

    VkBool32 supported = VK_FALSE;
    VkResult result = vpGetPhysicalDeviceProfileSupport(
        **instance,
        **physicalDevice,
        &appInfo.profile,
        &supported
    );

    if (result == VK_SUCCESS && supported == VK_TRUE) {
        appInfo.profileSupported = true;
        std::println("Using KHR roadmap 2022 profile");
    } else {
        appInfo.profileSupported = false;
        std::println("Fall back to traditional rendering (profile not supported)");

        detectFeatureSupport();
    }
}

void SoEngine::detectFeatureSupport() {
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

void SoEngine::createLogicalDevice() {
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

void SoEngine::createMemoryAllocator() {
    VmaAllocatorCreateInfo allocatorInfo {
        .flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
        .physicalDevice = **physicalDevice,
        .device = *device,
        .instance = **instance,
    };
    if (vmaCreateAllocator(&allocatorInfo, &allocator) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create VMA allocator");
    }

    mainDeletionQueue.pushFunction([this]() {
        vmaDestroyAllocator(allocator);
    });
}

void SoEngine::createSwapChain() {
    auto surfaceCapabilities = physicalDevice->getSurfaceCapabilitiesKHR(*surface);
    swapChainImageFormat = chooseSwapSurfaceFormat(
        physicalDevice->getSurfaceFormatsKHR(*surface)
    );
    if (surfaceCapabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
        swapChainExtent = surfaceCapabilities.currentExtent;
    }
    // else the surface size is undefined on some platforms
    // use default extent as set before
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
}

void SoEngine::createImageViews() {
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

void SoEngine::createRenderPass() {
    if (appInfo.profileSupported) {
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

void SoEngine::createFramebuffers() {
    if (appInfo.profileSupported) {
        // No framebuffers needed with dynamic rendering
        std::println("Using dynamic rendering, skipping framebuffer creation.");
        return;
    }

    std::println("Creating traditional framebuffers");

    swapChainFramebuffers.clear();
    for (size_t i = 0; i < swapChainImageViews.size(); ++i) {
        std::array attachments {
            colorImage.imageView,
            depthImage.imageView,
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

void SoEngine::createDescriptorSetLayout() {
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

void SoEngine::createGraphicsPipeline() {
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
    if (appInfo.profileSupported || appInfo.dynamicRenderingSupported) {
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

void SoEngine::createCommandPool() {
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

void SoEngine::createColorResources() {
    vk::Format colorFormat = swapChainImageFormat;
    colorImage = createAllocatedImage(device, allocator,
        {swapChainExtent.width, swapChainExtent.height, 1},
        1, msaaSamples,
        colorFormat, vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eTransientAttachment | vk::ImageUsageFlagBits::eColorAttachment,
        vk::MemoryPropertyFlagBits::eDeviceLocal);

    resourceDeletionQueue.pushFunction([=, this]() {
        (*device).destroyImageView(colorImage.imageView);
        vmaDestroyImage(allocator, colorImage.image, colorImage.allocation);
    });
}

void SoEngine::createDepthResources() {
    vk::Format depthFormat = findDepthFormat();
    depthImage = createAllocatedImage(device, allocator,
        {swapChainExtent.width, swapChainExtent.height, 1},
        1, msaaSamples,
        depthFormat, vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eDepthStencilAttachment,
        vk::MemoryPropertyFlagBits::eDeviceLocal
    );

    resourceDeletionQueue.pushFunction([=, this]() {
        (*device).destroyImageView(depthImage.imageView);
        vmaDestroyImage(allocator, depthImage.image, depthImage.allocation);
    });
}

void SoEngine::createTextureImage() {
    // Load KTX2 texture
    ktxVulkanDeviceInfo* vdi = ktxVulkanDeviceInfo_Create(
        **physicalDevice, *device,
        *transferQueue, *transferCommandPool, nullptr);

    ktxTexture2* kTexture;
    KTX_error_code ktxResult = ktxTexture2_CreateFromNamedFile(
        (appDir / TEXTURE_PATH).c_str(),
        KTX_TEXTURE_CREATE_NO_FLAGS,
        &kTexture);
    if (ktxResult != KTX_SUCCESS) {
        throw std::runtime_error(std::format("Failed to load ktx texture image \"{}\"", TEXTURE_PATH));
    }
    mipLevels = kTexture->numLevels;
    kTexture->vkFormat = VK_FORMAT_R8G8B8A8_SRGB;

    ktxResult = ktxTexture2_VkUploadEx(kTexture, vdi, &texture,
        VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    switch (ktxResult) {
        case KTX_SUCCESS:
            // Success, no action needed.
            break;
        case KTX_INVALID_VALUE:
            throw std::runtime_error("KTX error: Invalid value (possibly incomplete parameters or callbacks)");
        case KTX_INVALID_OPERATION:
            throw std::runtime_error("KTX error: Invalid operation (unsupported format, tiling, usageFlags, mipmap generation, or too many mip levels/layers)");
        case KTX_OUT_OF_MEMORY:
            throw std::runtime_error("KTX error: Out of memory (insufficient memory on CPU or Vulkan device)");
        case KTX_UNSUPPORTED_FEATURE:
            throw std::runtime_error("KTX error: Unsupported feature (sparse binding of KTX textures is not supported)");
        default:
            throw std::runtime_error("KTX error: Unknown error code");
    }

    ktxTexture_Destroy(ktxTexture(kTexture));
    ktxVulkanDeviceInfo_Destroy(vdi);

    mainDeletionQueue.pushFunction([=, this]() {
        ktxVulkanTexture_Destruct(&texture, *device, nullptr);
    });
}

void SoEngine::createTextureImageView() {
    textureImageView = createImageView(device, texture.image,
        static_cast<vk::Format>(texture.imageFormat), vk::ImageAspectFlagBits::eColor, mipLevels);
}

void SoEngine::createTextureSampler() {
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

void SoEngine::loadModel(const std::string& modelName) {
    std::filesystem::path modelDir {appDir / "models" / modelName};
    std::filesystem::path modelPath {};
    for (const auto& entry : std::filesystem::directory_iterator(modelDir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".gltf") {
            if (modelPath.empty()) {
                modelPath = entry.path();
            } else {
                throw std::runtime_error(std::format("Directory \"{}\" has more than one glTF file.", modelName));
            }
        }
    }
    if (modelPath.empty()) {
        throw std::runtime_error(std::format("Directory \"{}\" has no glTF file.", modelName));
    }

    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err, warn;
    bool ret = loader.LoadASCIIFromFile(&model, &err, &warn, modelPath.string());

    if (!warn.empty()) {
        std::println("glTF warning: {}", warn);
    }
    if (!err.empty()) {
        std::println("glTF error: {}", err);
    }
    if (!ret) {
        throw std::runtime_error(std::format("Failed to load glTF model \"{}\"", modelName));
    }


    std::unordered_map<Vertex, uint32_t> uniqueVertices {};

    for (const auto& mesh : model.meshes) {
        for (const auto& primitive : mesh.primitives) {
            // Get indices
            const tinygltf::Accessor& indexAccessor = model.accessors[primitive.indices];
            const tinygltf::BufferView& indexBufferView = model.bufferViews[indexAccessor.bufferView];
            const tinygltf::Buffer& indexBuffer = model.buffers[indexBufferView.buffer];

            // Get vertex positions
            const tinygltf::Accessor& posAccessor = model.accessors[primitive.attributes.at("POSITION")];
            const tinygltf::BufferView& posBufferView = model.bufferViews[posAccessor.bufferView];
            const tinygltf::Buffer& posBuffer = model.buffers[posBufferView.buffer];

            // Get texture coordinates if available
            bool hasTexCoords = primitive.attributes.find("TEXCOORD_0") != primitive.attributes.end();
            const tinygltf::Accessor* texCoordAccessor = nullptr;
            const tinygltf::BufferView* texCoordBufferView = nullptr;
            const tinygltf::Buffer* texCoordBuffer = nullptr;

            if (hasTexCoords) {
                texCoordAccessor = &model.accessors[primitive.attributes.at("TEXCOORD_0")];
                texCoordBufferView = &model.bufferViews[texCoordAccessor->bufferView];
                texCoordBuffer = &model.buffers[texCoordBufferView->buffer];
            }

            uint32_t baseVertex = static_cast<uint32_t>(vertices.size());

            // Process vertices
            for (size_t i = 0; i < posAccessor.count; i++) {
                Vertex vertex{};

                // Get position
                const float* pos = reinterpret_cast<const float*>(&posBuffer.data[posBufferView.byteOffset + posAccessor.byteOffset + i * 12]);
                vertex.pos = {pos[0], pos[1], pos[2]};
                maxX = std::max(maxX, pos[0]);
                minX = std::min(minX, pos[0]);
                maxY = std::max(maxY, pos[1]);
                minY = std::min(minY, pos[1]);
                maxZ = std::max(maxZ, pos[2]);
                minZ = std::min(minZ, pos[2]);

                // Get texture coordinates if available
                if (hasTexCoords) {
                    const float* texCoord = reinterpret_cast<const float*>(&texCoordBuffer->data[texCoordBufferView->byteOffset + texCoordAccessor->byteOffset + i * 8]);
                    vertex.texCoord = {texCoord[0], texCoord[1]};
                } else {
                    vertex.texCoord = {0.0f, 0.0f};
                }

                // Set default color
                vertex.color = {1.0f, 1.0f, 1.0f};

                vertices.emplace_back(std::move(vertex));
            }

            // Process indices
            const unsigned char* indexData = &indexBuffer.data[indexBufferView.byteOffset + indexAccessor.byteOffset];
            size_t indexCount = indexAccessor.count;
            size_t indexStride = 0;

            // Determine index stride based on component type
            if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
                indexStride = sizeof(uint16_t);
            } else if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
                indexStride = sizeof(uint32_t);
            } else if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE) {
                indexStride = sizeof(uint8_t);
            } else {
                throw std::runtime_error("Unsupported index component type");
            }

            indices.reserve(indices.size() + indexCount);

            for (size_t i = 0; i < indexCount; ++i) {
                uint32_t index = 0;

                if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
                    index = *reinterpret_cast<const uint16_t*>(indexData + i * indexStride);
                } else if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
                    index = *reinterpret_cast<const uint32_t*>(indexData + i * indexStride);
                } else if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE) {
                    index = *reinterpret_cast<const uint8_t*>(indexData + i * indexStride);
                }

                indices.emplace_back(baseVertex + index);
            }
        }
        std::println("vertices size: {}, indices size: {}", vertices.size(), indices.size());
    }
}

void SoEngine::setupModelObjects() {
    // Object 1 - Center
    modelObjects[0].position = {0.0f, 0.0f, 0.0f};
    modelObjects[0].rotation = {0.0f, 0.0f, 0.0f};
    modelObjects[0].scale = {1.0f, 1.0f, 1.0f};

    // Object 2 - Left
    modelObjects[1].position = {-2.0f, 0.0f, -1.0f};
    modelObjects[1].rotation = {0.0f, glm::radians(45.0f), 0.0f};
    modelObjects[1].scale = {0.75f, 0.75f, 0.75f};

    // Object 3 - Right
    modelObjects[2].position = {2.0f, 0.0f, -1.0f};
    modelObjects[2].rotation = {0.0f, glm::radians(-45.0f), 0.0f};
    modelObjects[2].scale = {0.75f, 0.75f, 0.75f};
}

void SoEngine::createVertexBuffer() {
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

void SoEngine::createIndexBuffer() {
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

void SoEngine::createUniformBuffers() {
    // For each model object
    for (auto& object : modelObjects) {
        object.uniformBuffers.clear();
        object.uniformBuffersMemory.clear();
        object.uniformBuffersMapped.clear();

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
            constexpr vk::DeviceSize bufferSize = sizeof(UniformBufferObject);
            vk::raii::Buffer buffer {{}};
            vk::raii::DeviceMemory bufferMemory {{}};
            createBuffer(bufferSize, vk::BufferUsageFlagBits::eUniformBuffer,
                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                buffer, bufferMemory
            );
            object.uniformBuffers.emplace_back(std::move(buffer));
            object.uniformBuffersMemory.emplace_back(std::move(bufferMemory));
            object.uniformBuffersMapped.emplace_back(object.uniformBuffersMemory[i].mapMemory(0, bufferSize));
        }
    }
}

void SoEngine::createDescriptorPool() {
    std::array poolSizes {
        vk::DescriptorPoolSize{
            .type = vk::DescriptorType::eUniformBuffer,
            .descriptorCount = MAX_OBJECTS * MAX_FRAMES_IN_FLIGHT
        },
        vk::DescriptorPoolSize{
            .type = vk::DescriptorType::eCombinedImageSampler,
            .descriptorCount = MAX_OBJECTS * MAX_FRAMES_IN_FLIGHT
        }
    };

    vk::DescriptorPoolCreateInfo poolInfo {
        .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
        .maxSets = MAX_OBJECTS * MAX_FRAMES_IN_FLIGHT,
        .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
        .pPoolSizes = poolSizes.data(),
    };

    descriptorPool = vk::raii::DescriptorPool{device, poolInfo};
}

void SoEngine::createDescriptorSets() {
    // For each model object
    for (auto& object : modelObjects) {
        // Create descriptor sets for each frame in flight
        std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *descriptorSetLayout);
        vk::DescriptorSetAllocateInfo allocInfo {
            .descriptorPool = *descriptorPool,
            .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
            .pSetLayouts = layouts.data()
        };

        object.descriptorSets.clear();
        object.descriptorSets = device.allocateDescriptorSets(allocInfo);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
            vk::DescriptorBufferInfo bufferInfo {
                .buffer = *object.uniformBuffers[i],
                .offset = 0,
                .range = sizeof(UniformBufferObject)
            };

            vk::DescriptorImageInfo imageInfo {
                .sampler = *textureSampler,
                .imageView = *textureImageView,
                .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
            };

            std::array descriptorWrites {
                vk::WriteDescriptorSet{
                    .dstSet = *object.descriptorSets[i],
                    .dstBinding = 0,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = vk::DescriptorType::eUniformBuffer,
                    .pBufferInfo = &bufferInfo,
                },
                vk::WriteDescriptorSet{
                    .dstSet = *object.descriptorSets[i],
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
}

void SoEngine::createFrameData() {
    frames.clear();

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        frames.emplace_back(device, graphicsCommandPool);
    }
}

void SoEngine::recordCommandBuffer(uint32_t imageIndex) {
    getCurrentFrame().commandBuffer.begin({});

    vk::ClearValue clearColor = vk::ClearColorValue{0.0f, 0.0f, 0.0f, 1.0f};
    vk::ClearValue clearDepth = vk::ClearDepthStencilValue(1.0f, 0);
    vk::ClearValue clearResolve = vk::ClearColorValue{0.0f, 0.0f, 0.0f, 1.0f};
    std::array clearValues = { clearColor, clearDepth, clearResolve};

    if (appInfo.profileSupported || appInfo.dynamicRenderingSupported) {
        // begin dynamic rendering
        auto commandBuffer = beginSingleTimeCommands(graphicsCommandPool);
        transitionImageLayout(commandBuffer,
            swapChainImages[imageIndex],
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::eColorAttachmentOptimal);
        // multisampled color image
        transitionImageLayout(commandBuffer,
            colorImage.image,
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::eColorAttachmentOptimal);
        // depth image
        transitionImageLayout(commandBuffer,
            depthImage.image,
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::eDepthAttachmentOptimal);
        endSingleTimeCommands(commandBuffer, graphicsQueue);

        vk::RenderingAttachmentInfo colorAttachmentInfo {
            .imageView = colorImage.imageView,
            .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .resolveMode = vk::ResolveModeFlagBits::eAverage,
            .resolveImageView = swapChainImageViews[imageIndex],
            .resolveImageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .clearValue = clearColor,
        };
        vk::RenderingAttachmentInfo depthAttachmentInfo {
            .imageView = depthImage.imageView,
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

        getCurrentFrame().commandBuffer.beginRendering(renderingInfo);
    } else {
        // Use traditional render pass
        vk::RenderPassBeginInfo renderPassBeginInfo {
            .renderPass = *renderPass,
            .framebuffer = *swapChainFramebuffers[imageIndex],
            .renderArea = {{0, 0}, swapChainExtent},
            .clearValueCount = static_cast<uint32_t>(clearValues.size()),
            .pClearValues = clearValues.data(),
        };

        getCurrentFrame().commandBuffer.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);
    }

    getCurrentFrame().commandBuffer.setViewport(0,
        vk::Viewport{
            0.0f, 0.0f,
            static_cast<float>(swapChainExtent.width),
            static_cast<float>(swapChainExtent.height),
            0.0f, 1.0f
        });
    getCurrentFrame().commandBuffer.setScissor(0, vk::Rect2D{vk::Offset2D{0, 0}, swapChainExtent});
    getCurrentFrame().commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *graphicsPipeline);
    getCurrentFrame().commandBuffer.bindVertexBuffers(0, {*vertexBuffer}, {0});
    getCurrentFrame().commandBuffer.bindIndexBuffer(*indexBuffer, 0, vk::IndexType::eUint32);
    // Draw each object with its own descriptor set
    for (const auto& object : modelObjects) {
        getCurrentFrame().commandBuffer.bindDescriptorSets(
            vk::PipelineBindPoint::eGraphics,
            *pipelineLayout, 0,
            {*object.descriptorSets[currentFrame]}, {});
        // Draw the object
        getCurrentFrame().commandBuffer.drawIndexed(static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);
    }

    if (appInfo.profileSupported || appInfo.dynamicRenderingSupported) {
        getCurrentFrame().commandBuffer.endRendering();

        auto commandBuffer = beginSingleTimeCommands(graphicsCommandPool);
        transitionImageLayout(commandBuffer,
            swapChainImages[imageIndex],
            vk::ImageLayout::eColorAttachmentOptimal,
            vk::ImageLayout::ePresentSrcKHR);
        endSingleTimeCommands(commandBuffer, graphicsQueue);
    } else {
        getCurrentFrame().commandBuffer.endRenderPass();
    }

    getCurrentFrame().commandBuffer.end();
}

void SoEngine::updateUniformBuffer(uint32_t currentImage) {
    static auto startTime = std::chrono::high_resolution_clock::now();

    const auto currentTime = std::chrono::high_resolution_clock::now();
    const float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

    // glm::vec3 objCenter {(minX + maxX) / 2.0f, (minY + maxY) / 2.0f, (minZ + maxZ) / 2.0f };
    float xLen { maxX - minX }, yLen { maxY - minY }, zLen { maxZ - minZ };

    glm::vec3 eye { 0.0f, yLen * 2.5, zLen * 2.5};
    glm::vec3 up {0, 1, 0};
    glm::mat4 view = glm::lookAt(eye, glm::vec3(0.0f), up);
    glm::mat4 proj = glm::perspective(glm::radians(45.0f),
            static_cast<float>(swapChainExtent.width) / static_cast<float>(swapChainExtent.height),
            0.01f,
            static_cast<float>(glm::length(glm::vec3{xLen, yLen, zLen})) * 3);
    // GLM was designed for OpenGL
    // where Y coordinate of the clip coordinates is inverted
    proj[1][1] *= -1;

    // Update uniform buffers for each object
    for (auto& object : modelObjects) {
        glm::mat4 rotation = glm::rotate(glm::mat4(1.0f), time * glm::radians(30.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        glm::mat4 model = object.getModelMatrix() * rotation;

        UniformBufferObject ubo {
            .model = model,
            .view = view,
            .proj = proj
        };

        memcpy(object.uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
    }
}

void SoEngine::drawFrame() {
    while (vk::Result::eTimeout == device.waitForFences(*getCurrentFrame().inFlightFence, vk::True, UINT64_MAX))
        ;
    getCurrentFrame().deletionQueue.flush();

    // Acquire an image from the swap chain
    auto [result, imageIndex] = swapChain.acquireNextImage(
        UINT64_MAX, getCurrentFrame().presentCompleteSemaphore, nullptr);
    if (result == vk::Result::eErrorOutOfDateKHR) {
        swapChainOutOfDate = true;
        return;
    } else if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR) {
        throw std::runtime_error("Failed to acquire swap chain image!");
    }

    updateUniformBuffer(currentFrame);
    // Only reset the fence if we are submitting work
    device.resetFences(*getCurrentFrame().inFlightFence);

    // Record the command buffer
    getCurrentFrame().commandBuffer.reset();
    recordCommandBuffer(imageIndex);

    // Submit the command buffer
    // Traditional binary semaphores
    vk::PipelineStageFlags waitDestinationStageMask {
        vk::PipelineStageFlagBits::eColorAttachmentOutput
    };
    const vk::SubmitInfo submitInfo {
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &*getCurrentFrame().presentCompleteSemaphore,
        .pWaitDstStageMask = &waitDestinationStageMask,
        .commandBufferCount = 1,
        .pCommandBuffers = &*getCurrentFrame().commandBuffer,
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &*getCurrentFrame().renderFinishedSemaphore,
    };

    graphicsQueue.submit(submitInfo, *getCurrentFrame().inFlightFence);

    // Presentation
    const vk::PresentInfoKHR presentInfo {
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &*getCurrentFrame().renderFinishedSemaphore,
        .swapchainCount = 1,
        .pSwapchains = &*swapChain,
        .pImageIndices = &imageIndex,
        .pResults = nullptr, // To check multiple swap chains' results
    };
    result = presentQueue.presentKHR(presentInfo);
    if (result == vk::Result::eErrorOutOfDateKHR ||
        result == vk::Result::eSuboptimalKHR) {
        recreateSwapChain();
    } else if (result != vk::Result::eSuccess) {
        throw std::runtime_error("Failed to present swap chain image!");
    }

    currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}

void SoEngine::recreateSwapChain() {
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
bool SoEngine::isDeviceSuitable(const vk::raii::PhysicalDevice& device) {
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

std::vector<const char*> SoEngine::getRequiredLayers() const {
    std::vector<const char*> requiredLayers {};
    checkLayerSupport(requiredLayers);
    return requiredLayers;
}

void SoEngine::checkLayerSupport(const std::vector<const char*>& requiredLayers) const {
    const auto layerProperties = context.enumerateInstanceLayerProperties();
    if (std::ranges::any_of(requiredLayers, [&layerProperties](const auto& requiredLayer) {
        return std::ranges::none_of(layerProperties, [&requiredLayer](const auto& layerProperty) {
            return strcmp(requiredLayer, layerProperty.layerName) == 0;
        });
    })) {
        throw std::runtime_error("One or more required layer are not supported!");
    }
}

std::vector<const char*> SoEngine::getRequiredExtensions() const {
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

void SoEngine::checkExtensionSupport(const std::vector<const char*>& requiredExtensions) const {
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

uint32_t SoEngine::rateDeviceSuitability(const vk::raii::PhysicalDevice& device) {
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

QueueFamilyIndices SoEngine::findQueueFamilies(const vk::raii::PhysicalDevice& physicalDevice) const {
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

uint32_t SoEngine::findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) const {
    vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice->getMemoryProperties();

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; ++i) {
        if (typeFilter & (1 << i) &&
            (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
            }
    }

    throw std::runtime_error("Failed to find suitable memory type!");
}

vk::Format SoEngine::findSupportedFormat(
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

vk::Format SoEngine::findDepthFormat() const {
    return findSupportedFormat(
        { vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint},
        vk::ImageTiling::eOptimal,
        vk::FormatFeatureFlagBits::eDepthStencilAttachment
    );
}

vk::SampleCountFlagBits SoEngine::getMaxUsableSampleCount() const {
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

vk::Format SoEngine::chooseSwapSurfaceFormat(
    const std::vector<vk::SurfaceFormatKHR>& availableFormats) const noexcept
{
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

vk::PresentModeKHR SoEngine::chooseSwapPresentMode(
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

vk::raii::ShaderModule SoEngine::createShaderModule(const std::vector<char>& code) const{
    vk::ShaderModuleCreateInfo createInfo {
        .codeSize = code.size() * sizeof(char),
        .pCode = reinterpret_cast<const uint32_t*>(code.data())
    };

    return vk::raii::ShaderModule{device, createInfo};
}


void SoEngine::createBuffer(
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

vk::raii::CommandBuffer SoEngine::beginSingleTimeCommands(const vk::raii::CommandPool& commandPool) const {
    vk::CommandBufferAllocateInfo allocInfo {
        .commandPool = commandPool,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = 1
    };

    vk::raii::CommandBuffer commandBuffer{std::move(device.allocateCommandBuffers(allocInfo).front())};
    commandBuffer.begin(vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    return commandBuffer;
}

void SoEngine::endSingleTimeCommands(
    const vk::raii::CommandBuffer& commandBuffer,
    const vk::raii::Queue& queue
) const {
    commandBuffer.end();

    queue.submit(vk::SubmitInfo{.commandBufferCount = 1, .pCommandBuffers = &*commandBuffer}, nullptr);
    queue.waitIdle();
}

void SoEngine::copyBuffer(
    const vk::raii::Buffer& srcBuffer,
    const vk::raii::Buffer& dstBuffer,
    vk::DeviceSize size
) const {
    auto commandCopyBuffer = beginSingleTimeCommands(transferCommandPool);
    commandCopyBuffer.copyBuffer(srcBuffer, dstBuffer, vk::BufferCopy{0, 0, size});
    endSingleTimeCommands(commandCopyBuffer, transferQueue);
}



void SoEngine::copyBufferToImage(
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

void SoEngine::generateMipmaps(
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