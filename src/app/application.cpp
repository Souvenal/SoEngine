#include "application.h"

#include "render/vulkan/vk_utils.h"

#include <map>
#include <fstream>
#include <ranges>
#include <algorithm>

namespace {

VKAPI_ATTR vk::Bool32 VKAPI_CALL debugCallback(
    vk::DebugUtilsMessageSeverityFlagBitsEXT severity,
    vk::DebugUtilsMessageTypeFlagsEXT messageType,
    const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData) {
    if (severity >= vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning) {
        // std::println(std::cerr, "Validation layer: type {} msg: {}\n", to_string(messageType), pCallbackData->pMessage);
    }
    // indicate if the call that triggered this callback should be aborted
    return vk::False;
}

}

void AppInfo::checkFeatureSupport(
        vk::Instance instance,
        vk::PhysicalDevice physicalDevice) {
    // VkBool32 supported = VK_FALSE;
    // VkResult result = vpGetPhysicalDeviceProfileSupport(
    //     instance,
    //     physicalDevice,
    //     &profile,
    //     &supported
    // );

    // if (result == VK_SUCCESS && supported == VK_TRUE) {
    //     profileSupported = true;
    //     // std::println("Using KHR roadmap 2022 profile");
    // } else {
    //     profileSupported = false;
    //     // std::println("Fall back to traditional rendering (profile not supported)");

    detectFeatureSupport(physicalDevice);
    // }
}

void AppInfo::detectFeatureSupport(vk::PhysicalDevice physicalDevice) {
    auto deviceProperties = physicalDevice.getProperties();
    auto availableExtensions = physicalDevice.enumerateDeviceExtensionProperties();

    // Check for dynamic rendering support
    if (deviceProperties.apiVersion >= vk::ApiVersion13) {
        dynamicRenderingSupported = true;
        // std::println("Dynamic rendering supported via Vulkan 1.3");
    } else if (std::ranges::any_of(availableExtensions,
        [](const auto& availableExtension) {
            return strcmp(availableExtension.extensionName, vk::KHRDynamicRenderingExtensionName) == 0;
        })) {
        dynamicRenderingSupported = true;
        // std::println("Dynamic rendering supported via extension");
    }

    // Check for timeline semaphores support
    if (deviceProperties.apiVersion >= vk::ApiVersion12) {
        timelineSemaphoresSupported = true;
        // std::println("Timeline semaphores supported via Vulkan 1.2");
    } else if (std::ranges::any_of(availableExtensions,
        [](const auto& availableExtension) {
            return strcmp(availableExtension.extensionName, vk::KHRTimelineSemaphoreExtensionName) == 0;
        })) {
        timelineSemaphoresSupported = true;
        // std::println("Timeline semaphores supported via extension");
    }

    // Check for synchronization2 support
    if (deviceProperties.apiVersion >= vk::ApiVersion13) {
        synchronization2Supported = true;
        // std::println("Synchronization2 supported via Vulkan 1.3");
    } else if (std::ranges::any_of(availableExtensions,
        [](const auto& availableExtension) {
            return strcmp(availableExtension.extensionName, vk::KHRSynchronization2ExtensionName) == 0;
        })) {
        synchronization2Supported = true;
        // std::println("Synchronization2 supported via extension");
    }

    if (dynamicRenderingSupported && deviceProperties.apiVersion < vk::ApiVersion13) {
        requiredDeviceExtensions.emplace_back(vk::KHRDynamicRenderingExtensionName);
    }
    if (timelineSemaphoresSupported && deviceProperties.apiVersion < vk::ApiVersion12) {
        requiredDeviceExtensions.emplace_back(vk::KHRTimelineSemaphoreExtensionName);
    }
    if (synchronization2Supported && deviceProperties.apiVersion < vk::ApiVersion13) {
        requiredDeviceExtensions.emplace_back(vk::KHRSynchronization2ExtensionName);
    }
}

Application::Application(const std::filesystem::path& appDir):
    Application(appDir, AppInfo{}) { }

Application::Application(const std::filesystem::path& appDir, const AppInfo& appInfo):
    appDir(appDir), appInfo(appInfo) { }

Application::~Application() { }

void Application::onUpdate(double deltaTime) {
}

void Application::onRender() {
    drawFrame();
}

void Application::onInit(const Window* window) {
    this->window = window;

    initInstance("Default Application");
    initDebugMessenger();
    initSurface();
    selectPhysicalDevice();
    initLogicalDevice();

    initMemoryAllocator();
    initSwapchain();

    initCommandPools();
    // initCommandBuffers();

    initDescriptorAllocator();
    initGraphicsPipeline();

    initSyncObjects();

    initImGui();
}

void Application::onInputEvent(const InputEvent& event) {
    if (event.getSource() == EventSource::Mouse) {
        const auto& mouseButton = static_cast<const MouseButtonInputEvent&>(event);
    } else if (event.getSource() == EventSource::Keyboard) {
        const auto& keyEvent = static_cast<const KeyInputEvent&>(event);
    }
}

void Application::onShutdown() {
    if (device) {
        device->waitIdle();
    }

    cleanup();
}

bool Application::shouldTerminate() const {
    return false;
}

bool Application::isSwapchainOutOfDate() const {
    return swapchainOutOfDate;
}

void Application::recreateSwapchain() {
    ASSERT(device, "Logical device must be created before recreating swapchain");
    device->waitIdle();
    cleanupSwapchain();

    initSwapchain();
    // createFramebuffers();
}

void Application::initInstance(const std::string& appName) {
    vk::ApplicationInfo appInfo {
        .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
        .pEngineName = "So Engine",
        .engineVersion = VK_MAKE_VERSION(1, 0, 0),
        .apiVersion = vk::ApiVersion14
    };
    appInfo.pApplicationName = appName.c_str();

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
    createInfo.flags |= vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR;
#endif

    instance = std::make_unique<vk::raii::Instance>(context, createInfo);

    LOG_CORE_DEBUG("Instance is successfully initialized");
}

void Application::initDebugMessenger() {
#ifndef NDEBUG
    return;
#endif
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
    debugMessenger = instance->createDebugUtilsMessengerEXT(debugUtilsMessengerCreateInfoEXT);

    LOG_CORE_DEBUG("Debug messenger is successfully initialized");
}

void Application::initSurface() {
    ASSERT(window, "Window must be initialized before creating surface");

    surface = window->createSurface(*instance);
    auto framebufferSize = window->getFramebufferSize();
    swapchainExtent = vk::Extent2D { framebufferSize.width, framebufferSize.height };
    windowExtent = window->getWindowSize();

    LOG_CORE_DEBUG("Vulkan surface is successfully created");
}

void Application::selectPhysicalDevice() {
    auto physicalDevices = instance->enumeratePhysicalDevices();
    if (physicalDevices.empty()) {
        throw std::runtime_error("Failed to GPUs with Vulkan support!");
    }

    // Use an ordered map to automatically sort candidates by increasing score
    std::multimap<uint32_t, vk::raii::PhysicalDevice> candidates;
    for (const auto& physicalDevice : physicalDevices) {
        uint32_t score = vkutil::rateDeviceSuitability(
            physicalDevice, surface,
            appInfo.requiredDeviceExtensions);
        candidates.insert(std::make_pair(score, physicalDevice));
    }

    // Check if the best candidate is suitable at all
    if (candidates.rbegin()->first == 0) {
        throw std::runtime_error("Failed to find a suitable GPU!");
    }
    
    physicalDevice = std::make_unique<vk::raii::PhysicalDevice>(candidates.rbegin()->second);

    appInfo.msaaSamples = vkutil::getMaxUsableSampleCount(*physicalDevice);

    LOG_CORE_DEBUG("Physical device is successfully selected");
}

void Application::initLogicalDevice() {
    vk::StructureChain<
        vk::PhysicalDeviceFeatures2,
        vk::PhysicalDeviceVulkan13Features,
        vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT,
        vk::PhysicalDeviceTimelineSemaphoreFeaturesKHR,     // Timeline semaphores
        vk::PhysicalDeviceBufferDeviceAddressFeatures    // Buffer device address (vma)
    > featureChain {
        {.features = { .sampleRateShading = vk::True, .samplerAnisotropy = vk::True}},
        {.synchronization2 =  vk::True, .dynamicRendering = true},
        {.extendedDynamicState = true},
        {.timelineSemaphore = true},
        {.bufferDeviceAddress = true}
    };

    initLogicalDevice(featureChain);
}

void Application::initMemoryAllocator() {
    memoryAllocator = MemoryAllocator(**instance, **physicalDevice, *device);

    LOG_CORE_DEBUG("Memory allocator is successfully initialized");
}

void Application::initSwapchain() {
    swapchainImageFormat = vkutil::chooseSurfaceFormat(
        physicalDevice->getSurfaceFormatsKHR(*surface)
    );
    auto surfaceCapabilities = physicalDevice->getSurfaceCapabilitiesKHR(*surface);
    if (surfaceCapabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
        swapchainExtent = surfaceCapabilities.currentExtent;
    }
    // else the surface size is undefined on some platforms
    // use default extent as set before
    minImageCountInSwapchain = std::max( 3u, surfaceCapabilities.minImageCount );
    minImageCountInSwapchain = ( surfaceCapabilities.maxImageCount > 0 && minImageCountInSwapchain > surfaceCapabilities.maxImageCount ) ?
        surfaceCapabilities.maxImageCount : minImageCountInSwapchain;

    vk::SwapchainCreateInfoKHR createInfo {
        .surface = surface,
        .minImageCount = minImageCountInSwapchain,
        .imageFormat = swapchainImageFormat,
        .imageColorSpace = vk::ColorSpaceKHR::eSrgbNonlinear,
        .imageExtent = swapchainExtent,
        .imageArrayLayers = 1,  // the amount of layers each image consists of, always 1 unless developing a stereoscopic 3D app
        .imageUsage = vk::ImageUsageFlagBits::eColorAttachment,   // operations using the images in the swap chain for
        .preTransform = surfaceCapabilities.currentTransform,
        .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
        .presentMode = vkutil::chooseSwapPresentMode(physicalDevice->getSurfacePresentModesKHR(*surface)),
        .clipped = vk::True,
        .oldSwapchain = nullptr // specify one when the window is resized,
    };

    // Specify how to handle swap chain images across multiple queue families
    uint32_t graphicsIndex = queueFamilyIndices.graphicsFamily.value();
    uint32_t presentIndex = queueFamilyIndices.presentFamily.value();
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

    swapchain = vk::raii::SwapchainKHR(*device, createInfo);
    swapchainImages = swapchain.getImages();

    // Create swap chain image views
    vk::ImageViewCreateInfo imageViewCreateInfo{
        .viewType = vk::ImageViewType::e2D,
        .format = swapchainImageFormat,
        .components = {
            vk::ComponentSwizzle::eIdentity,
            vk::ComponentSwizzle::eIdentity,
            vk::ComponentSwizzle::eIdentity,
            vk::ComponentSwizzle::eIdentity
        },
        .subresourceRange = {
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1
        }
    };
    for (const auto& image : swapchainImages) {
        imageViewCreateInfo.image = image;
        swapchainImageViews.emplace_back(*device, imageViewCreateInfo);
    }

    LOG_CORE_DEBUG("Swap chain is successfully created");

    // Create draw image based on window size
    // drawImage = AllocatedImage(
    //     *device, memoryAllocator.allocator,
    //     {windowExtent.width, windowExtent.height, 1},
    //     1, msaaSamples,
    //     vk::Format::eR16G16B16A16Sfloat,
    //     vk::ImageTiling::eOptimal,
    //     vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst |
    //     // vk::ImageUsageFlagBits::eStorage |
    //     vk::ImageUsageFlagBits::eColorAttachment,
    //     MemoryType::DeviceLocal
    // );
}

void Application::initCommandPools() {
    vk::CommandPoolCreateInfo graphicsCommandPoolInfo {
        .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
        .queueFamilyIndex = queueFamilyIndices.graphicsFamily.value()
    };
    graphicsCommandPool = vk::raii::CommandPool(*device, graphicsCommandPoolInfo);
    vk::CommandPoolCreateInfo computeCommandPoolInfo {
        .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
        .queueFamilyIndex = queueFamilyIndices.computeFamily.value()
    };
    computeCommandPool = vk::raii::CommandPool(*device, computeCommandPoolInfo);
    vk::CommandPoolCreateInfo transferCommandPoolInfo {
        .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
        .queueFamilyIndex = queueFamilyIndices.transferFamily.value()
    };
    transferCommandPool = vk::raii::CommandPool(*device, transferCommandPoolInfo);

    LOG_CORE_DEBUG("Command pools are successfully initialized");
}

void Application::initRenderPass() { }

void Application::initFramebuffers() { }

void Application::initDescriptorAllocator() {
    std::vector<DescriptorAllocator::PoolSizeRatio> sizes {
        { vk::DescriptorType::eCombinedImageSampler, IMGUI_IMPL_VULKAN_MINIMUM_IMAGE_SAMPLER_POOL_SIZE }
    };

    globalDescriptorAllocator = DescriptorAllocator(*device, appInfo.maxFramesInFlight, sizes);
}

void Application::initSyncObjects() {
    vk::SemaphoreTypeCreateInfo semaphoreTypeCreateInfo{
        .semaphoreType = vk::SemaphoreType::eTimeline,
        .initialValue = 0};
    vk::SemaphoreCreateInfo semaphoreInfo{
        .pNext = &semaphoreTypeCreateInfo};
    semaphore = vk::raii::Semaphore(*device, semaphoreInfo);
    timelineValue = 0;

    inFlightFences.clear();
    for (size_t i = 0; i < appInfo.maxFramesInFlight; ++i) {
        inFlightFences.emplace_back(device->createFence({}));
    }

    LOG_CORE_DEBUG("Synchronization objects are successfully initialized");
}

void Application::initImGui() {
    // this initializes the core structures of imgui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    // io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;         // Enable Docking
    io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;       // Enable Multi-Viewport / Platform Windows
    // io.ConfigViewportsNoAutoMerge = true;
    // io.ConfigViewportsNoTaskBarIcon = true;

    ImGui::StyleColorsDark();

    // this initializes imgui for glfw
    ImGui_ImplGlfw_InitForVulkan(window->getGLFWwindow(), true);
    ImGui_ImplVulkan_InitInfo initInfo {
        .Instance = **instance,
        .PhysicalDevice = **physicalDevice,
        .Device = **device,
        .QueueFamily = queueFamilyIndices.graphicsFamily.value(),
        .Queue = *graphicsQueue,
        .DescriptorPool = globalDescriptorAllocator.getPool(),
        .MinImageCount = minImageCountInSwapchain,
        .ImageCount = static_cast<uint32_t>(swapchainImages.size()),
        .MSAASamples = VkSampleCountFlagBits(vk::SampleCountFlagBits::e1),
        .UseDynamicRendering = true,
        .PipelineRenderingCreateInfo = pipelineRenderingCreateInfo,
        .CheckVkResultFn = [](VkResult err) {
            if (err != VK_SUCCESS) {
                LOG_CORE_ERROR("Vulkan error during ImGui initialization: {}", vk::to_string(vk::Result(err)));
            }
        }};

    ImGui_ImplVulkan_Init(&initInfo);

    // Load fonts
    // ImGui_ImplVulkan_CreateFontsTexture(commandBuffer);
    // submit
    // ImGui_ImplVulkan_DestroyFontUploadObjects();
}

void Application::buildCapabilitiesSummary() {
    auto props = physicalDevice->getProperties();
    capsSummary.gpuName.assign(
        props.deviceName.data(),
        strnlen(props.deviceName.data(), vk::MaxPhysicalDeviceNameSize)
    );
    capsSummary.apiVersionMajor = vk::apiVersionMajor(props.apiVersion);
    capsSummary.apiVersionMinor = vk::apiVersionMinor(props.apiVersion);
    capsSummary.apiVersionPatch = vk::apiVersionPatch(props.apiVersion);
    capsSummary.swapImageCount = static_cast<uint32_t>(swapchainImages.size());
    capsSummary.presentMode = vkutil::chooseSwapPresentMode(physicalDevice->getSurfacePresentModesKHR(*surface));
    capsSummary.swapFormat = swapchainImageFormat;
    capsSummary.dynamicRendering = appInfo.dynamicRenderingSupported;
    capsSummary.timelineSemaphores = appInfo.timelineSemaphoresSupported;
    capsSummary.sync2 = appInfo.synchronization2Supported;
    // capsSummary.profileSupported = appInfo.profileSupported;
}

void Application::logCapabilitiesSummary() const {
    LOG_CORE_INFO("Vulkan Device: {} (API {}.{}.{})",
        capsSummary.gpuName,
        capsSummary.apiVersionMajor,
        capsSummary.apiVersionMinor,
        capsSummary.apiVersionPatch);
    LOG_CORE_INFO("SwapImages={}\tPresentMode={}\tFormat={}",
        capsSummary.swapImageCount,
        vk::to_string(capsSummary.presentMode),
        vk::to_string(capsSummary.swapFormat));
    LOG_CORE_INFO("ProfileSupported={}\tDynamicRendering={}\tTimelineSemaphores={}\tSync2={}",
        capsSummary.profileSupported,
        capsSummary.dynamicRendering,
        capsSummary.timelineSemaphores,
        capsSummary.sync2);
}

void Application::cleanup() {
    cleanupSwapchain();
    mainDeletionQueue.flush();
}

void Application::cleanupSwapchain() {
    swapchainImageViews.clear();
    swapchain = nullptr;
    resourceDeletionQueue.flush();
}

/**
 * Helper functions
 */

std::vector<const char*> Application::getRequiredLayers() const {
    std::vector<const char*> requiredLayers {};
    checkLayerSupport(requiredLayers);
    return requiredLayers;
}

void Application::checkLayerSupport(const std::vector<const char*>& requiredLayers) const {
    const auto layerProperties = context.enumerateInstanceLayerProperties();
    if (std::ranges::any_of(requiredLayers, [&layerProperties](const auto& requiredLayer) {
        return std::ranges::none_of(layerProperties, [&requiredLayer](const auto& layerProperty) {
            return strcmp(requiredLayer, layerProperty.layerName) == 0;
        });
    })) {
        throw std::runtime_error("One or more required layer are not supported!");
    }
}

std::vector<const char*> Application::getRequiredExtensions() const {
    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

// In release mode, do not include debug utils extension
#ifdef NDEBUG
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
        LOG_CORE_WARN("VK_EXT_debug_utils extension not available. Validation layers may not work.");
    }
#endif

#ifdef __APPLE__
    extensions.emplace_back(vk::KHRPortabilityEnumerationExtensionName);
    // // needed by VK_KHR_portability_subset
    // extensions.emplace_back(vk::KHRGetPhysicalDeviceProperties2ExtensionName);
    // deprecated extension VK_KHR_get_physical_device_properties2, but this extension has been promoted to 1.1.0 (0x00401000)
#endif
    // for (const auto& extension : extensions) {
    //     LOG_CORE_DEBUG("Required instance extension: {}", extension);
    // }

    checkExtensionSupport(extensions);
    return extensions;
}

void Application::checkExtensionSupport(const std::vector<const char*>& requiredExtensions) const {
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

void Application::dumpAllocatedBuffer(
        const AllocatedBuffer& srcBuffer,
        vk::DeviceSize bufferSize,
        const std::string& filename) {
    ASSERT(device, "Invalid device");
    // 1. 创建 staging buffer
    AllocatedBuffer stagingBuffer{
        memoryAllocator.allocator, queueFamilyIndices, bufferSize,
        vk::BufferUsageFlagBits::eTransferDst,
        MemoryType::HostVisible
    };

    // 2. 拷贝数据到 staging buffer
    auto cmd = vkutil::beginSingleTimeCommands(*device, transferCommandPool);
    vkutil::copyAllocatedBuffer(cmd, srcBuffer, stagingBuffer, bufferSize);
    vkutil::endSingleTimeCommands(cmd, transferQueue);

    // 3. 映射内存
    void* data = stagingBuffer.map();
    // 4. 写入文件
    std::filesystem::path logPath = appDir / "logs" / filename;
    std::ofstream ofs(logPath, std::ios::binary);
    ofs.write(reinterpret_cast<const char*>(data), bufferSize);
    ofs.close();
    stagingBuffer.unmap();

    LOG_CORE_INFO("Buffer dumped to {}", logPath.string());
}