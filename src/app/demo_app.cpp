#include "demo_app.h"

#include "render/vulkan/vk_images.h"
#include "render/vulkan/vk_pipelines.h"
#include "render/vulkan/vk_initializers.h"

DemoApp::DemoApp(
        const std::filesystem::path& appDir,
        const Window* window,
        const AppInfo& appInfo,
        const DemoAppInfo& info)
    : Application(appDir, window, appInfo), demoAppInfo(info) { }

DemoApp::~DemoApp() { }

void DemoApp::onInit() {
    LOG_CORE_INFO("====Demo application initialization start====");

    initInstance("Demo Application");
    initDebugMessenger();
    initSurface();

    selectPhysicalDevice();
    appInfo.checkFeatureSupport(*instance, *physicalDevice);
    initLogicalDevice();

    initCommandPools();
    initCommandBuffers();

    initMemoryAllocator();
    initSwapchain();

    initDescriptorAllocator();
    initDescriptorSetLayouts();

    initPipelines();

    initDescriptorSets();

    initSyncObjects();

    initImGui();

    buildCapabilitiesSummary();
    logCapabilitiesSummary();

    LOG_CORE_INFO("====Demo application initialization done====");
}

void DemoApp::onShutdown() {
    Application::onShutdown();
}

void DemoApp::initSwapchain() {
    Application::initSwapchain();

    vk::Format drawImageFormat = vk::Format::eR16G16B16A16Sfloat;
    vk::Extent3D drawImageExtent = {windowExtent.width, windowExtent.height, 1};
    drawImage = AllocatedImage(
        *device, memoryAllocator.allocator,
        drawImageExtent, 1, vk::SampleCountFlagBits::e1,
        drawImageFormat, vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst,
        vk::ImageAspectFlagBits::eColor, MemoryType::DeviceLocal);

    LOG_CORE_DEBUG("Initialized swapchain and draw image with format: {}", vk::to_string(drawImageFormat));
}

void DemoApp::initCommandBuffers() {
    vk::CommandBufferAllocateInfo allocInfo{
        .commandPool = *graphicsCommandPool,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = static_cast<uint32_t>(appInfo.maxFramesInFlight)
    };
    graphicsCommandBuffers = vk::raii::CommandBuffers(*device, allocInfo);
    vk::CommandBufferAllocateInfo computeAllocInfo{
        .commandPool = *computeCommandPool,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = static_cast<uint32_t>(appInfo.maxFramesInFlight)
    };
    computeCommandBuffers = vk::raii::CommandBuffers(*device, computeAllocInfo);

    LOG_CORE_DEBUG("Initialized {} graphics and {} compute command buffers",
        graphicsCommandBuffers.size(), computeCommandBuffers.size());
}

void DemoApp::initDescriptorAllocator() {
    uint32_t imguiSets = appInfo.maxFramesInFlight;
    uint32_t ssboSets = appInfo.maxFramesInFlight;
    uint32_t maxSets = imguiSets + ssboSets;

    std::vector<DescriptorAllocator::PoolSizeRatio> sizes = {
        { vk::DescriptorType::eStorageImage, 1.0f },
        { vk::DescriptorType::eCombinedImageSampler, 1.0f }
    };
    globalDescriptorAllocator = DescriptorAllocator(*device, maxSets, sizes);

    LOG_CORE_DEBUG("Initialized descriptor allocator with max sets: {}", maxSets);
}

void DemoApp::initDescriptorSetLayouts() {
    DescriptorLayoutBuilder builder;
    builder.addBinding(0, vk::DescriptorType::eStorageImage, vk::ShaderStageFlagBits::eCompute);
    drawImageDescriptorSetLayout = builder.build(*device);

    LOG_CORE_DEBUG("Initialized descriptor set layouts");
}

void DemoApp::initPipelines() {
    initBackgroundPipelines();
}

void DemoApp::initBackgroundPipelines() {
    vk::PipelineLayoutCreateInfo computeLayout{
        .setLayoutCount = 1,
        .pSetLayouts = &*drawImageDescriptorSetLayout};
    gradientPipelineLayout = device->createPipelineLayout(computeLayout);

    auto shaderPath = appDir / "shaders" / "gradient.spv";
    auto computeDrawShader = vkutil::loadShaderModule(*device, shaderPath.string());

    vk::PipelineShaderStageCreateInfo computeStage{
        .stage = vk::ShaderStageFlagBits::eCompute,
        .module = computeDrawShader,
        .pName = "compMain"};

    vk::ComputePipelineCreateInfo computePipelineInfo{
        .stage = computeStage,
        .layout = gradientPipelineLayout};

    gradientPipeline = device->createComputePipeline(nullptr, computePipelineInfo);
}

void DemoApp::initDescriptorSets() {
    drawImageDescriptorSets = globalDescriptorAllocator.allocate(*device, drawImageDescriptorSetLayout);

    vk::DescriptorImageInfo imageInfo {
        .sampler = {},
        .imageView = drawImage.imageView,
        .imageLayout = vk::ImageLayout::eGeneral
    };

    vk::WriteDescriptorSet drawImageWrite {
        .dstSet = drawImageDescriptorSets[0],
        .dstBinding = 0,
        .descriptorCount = 1,
        .descriptorType = vk::DescriptorType::eStorageImage,
        .pImageInfo = &imageInfo
    };

    device->updateDescriptorSets(drawImageWrite, {});

    LOG_CORE_DEBUG("Initialized descriptor set for draw image");
}

void DemoApp::drawFrame() {
    // Acquire an image from the swap chain
    auto [result, imageIndex] = swapchain.acquireNextImage(
        UINT64_MAX, nullptr, inFlightFences[currentFrame]);
    while (vk::Result::eTimeout == device->waitForFences(
        {inFlightFences[currentFrame]}, vk::True,
        std::numeric_limits<uint64_t>::max()))
        ;
    
    if (result == vk::Result::eErrorOutOfDateKHR) {
        LOG_CORE_WARN("Swap chain is out of date during image acquisition. Recreating swap chain...");
        swapchainOutOfDate = true;
        return;
    } else if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR) {
        throw std::runtime_error("Failed to acquire swap chain image!");
    }

    // Only reset the fence if we are submitting work
    device->resetFences(*inFlightFences[currentFrame]);

    uint64_t computeWaitValue = timelineValue;
    uint64_t computeSignalValue = ++timelineValue;
    uint64_t graphicsWaitValue = timelineValue;
    uint64_t graphicsSignalValue = ++timelineValue;
    
    const auto& computeCommandBuffer = computeCommandBuffers[currentFrame];
    const auto& graphicsCommandBuffer = graphicsCommandBuffers[currentFrame];
    computeCommandBuffer.reset();
    computeCommandBuffer.begin({.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    graphicsCommandBuffer.reset();
    graphicsCommandBuffer.begin({.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    // Compute pass
    {
    //> Draw background using compute shader
        vkutil::transitionImageLayout(
            computeCommandBuffer, drawImage.image,
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::eGeneral);

        drawBackground(computeCommandBuffer);

        vkutil::transitionImageLayout(
            computeCommandBuffer, drawImage.image,
            vk::ImageLayout::eGeneral,
            vk::ImageLayout::eTransferSrcOptimal);

        computeCommandBuffer.end();
        auto commandBufferInfo = vkinit::commandBufferSubmitInfo(computeCommandBuffer);
        auto waitSubmitInfo = vkinit::semaphoreSubmitInfo(
            vk::PipelineStageFlagBits2::eTopOfPipe, *semaphore, computeWaitValue);
        auto signalSubmitInfo = vkinit::semaphoreSubmitInfo(
            vk::PipelineStageFlagBits2::eComputeShader, *semaphore, computeSignalValue);
        auto submitInfo = vkinit::submitInfo(
            commandBufferInfo, waitSubmitInfo, signalSubmitInfo);
        graphicsQueue.submit2(submitInfo, {});
    //<
    }
    // Graphics pass
    {
        vkutil::transitionImageLayout(
            graphicsCommandBuffer, swapchainImages[imageIndex],
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::eTransferDstOptimal);

        vk::Extent2D drawExtent = {drawImage.imageExtent.width, drawImage.imageExtent.height};
        vkutil::copyImageToImage(
            graphicsCommandBuffer,
            drawImage.image, swapchainImages[imageIndex],
            drawExtent, swapchainExtent);

        vkutil::transitionImageLayout(
            graphicsCommandBuffer, swapchainImages[imageIndex],
            vk::ImageLayout::eTransferDstOptimal,
            vk::ImageLayout::eColorAttachmentOptimal);

        // drawScene(commandBuffer, imageIndex);

        drawImGui(graphicsCommandBuffer, imageIndex);

        vkutil::transitionImageLayout(graphicsCommandBuffer,
            swapchainImages[imageIndex],
            vk::ImageLayout::eColorAttachmentOptimal,
            vk::ImageLayout::ePresentSrcKHR
        );

        graphicsCommandBuffer.end();

        auto commandBufferInfo = vkinit::commandBufferSubmitInfo(*graphicsCommandBuffers[currentFrame]);
        auto waitSubmitInfo = vkinit::semaphoreSubmitInfo(
            vk::PipelineStageFlagBits2::eTransfer, *semaphore, graphicsWaitValue);
        auto signalSubmitInfo = vkinit::semaphoreSubmitInfo(
            vk::PipelineStageFlagBits2::eColorAttachmentOutput, *semaphore, graphicsSignalValue);
        auto submitInfo = vkinit::submitInfo(
            commandBufferInfo, waitSubmitInfo, signalSubmitInfo);

        graphicsQueue.submit2(submitInfo, {});
    }

    vk::SemaphoreWaitInfo waitInfo {
        .semaphoreCount = 1,
        .pSemaphores = &*semaphore,
        .pValues = &graphicsSignalValue
    };
    while (vk::Result::eTimeout == device->waitSemaphores(waitInfo, std::numeric_limits<uint64_t>::max()))
        ;

    // Presentation
    const vk::PresentInfoKHR presentInfo {
        .swapchainCount = 1,
        .pSwapchains = &*swapchain,
        .pImageIndices = &imageIndex,
        .pResults = nullptr, // To check multiple swap chains' results
    };
    result = presentQueue.presentKHR(presentInfo);
    if (result == vk::Result::eErrorOutOfDateKHR ||
        result == vk::Result::eSuboptimalKHR) {
        LOG_CORE_WARN("Swap chain is out of date or suboptimal during presentation. Recreating swap chain...");
        recreateSwapchain();
    } else if (result != vk::Result::eSuccess) {
        throw std::runtime_error("Failed to present swap chain image!");
    }

    currentFrame = (currentFrame + 1) % appInfo.maxFramesInFlight;
} 

void DemoApp::drawBackground(const vk::raii::CommandBuffer& commandBuffer) {
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *gradientPipeline);
    commandBuffer.bindDescriptorSets(
        vk::PipelineBindPoint::eCompute, *gradientPipelineLayout,
        0, *drawImageDescriptorSets[0], {});
    commandBuffer.dispatch(
        std::ceil(drawImage.imageExtent.width / 16.0f),
        std::ceil(drawImage.imageExtent.height / 16.0f),
        1);
}