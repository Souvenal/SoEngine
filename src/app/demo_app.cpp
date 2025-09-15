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

    initDescriptorAllocator();
    initMemoryAllocator();

    initSwapchain();
    initDrawImage();

    initCommandPools();
    initCommandBuffers();

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
    Application::initDescriptorSetLayouts();
}

void DemoApp::initPipelines() {
    initBackgroundPipelines();
}

void DemoApp::initBackgroundPipelines() {
    auto shadersDir = appDir / "shaders";
    auto gradientShader = vkutil::loadShaderModule(*device, shadersDir / "gradient_color.spv");
    auto skyShader = vkutil::loadShaderModule(*device, shadersDir / "sky.spv");

    vk::PushConstantRange pushConstantRange{
        .stageFlags = vk::ShaderStageFlagBits::eCompute,
        .offset = 0,
        .size = sizeof(ComputePushConstants)};
    vk::PipelineLayoutCreateInfo computeLayout{
        .setLayoutCount = 1,
        .pSetLayouts = &*drawImageDescriptorSetLayout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pushConstantRange};
    gradientPipelineLayout = device->createPipelineLayout(computeLayout);

    // Create gradient effect
    vk::PipelineShaderStageCreateInfo stageInfo{
        .stage = vk::ShaderStageFlagBits::eCompute,
        .module = gradientShader,
        .pName = "compMain"};

    vk::ComputePipelineCreateInfo computePipelineInfo{
        .stage = stageInfo,
        .layout = gradientPipelineLayout};

    ComputeEffect gradientEffect;
    gradientEffect.name = "gradient";
    gradientEffect.layout = gradientPipelineLayout;
    gradientEffect.pipeline = device->createComputePipeline(nullptr, computePipelineInfo);
    gradientEffect.data = {
        .data1 = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f), // Top color
        .data2 = glm::vec4(0.0f, 0.0f, 1.0f, 1.0f), // Bottom color
        .data3 = glm::vec4(0.0f),
        .data4 = glm::vec4(0.0f)};

    // Create sky effect - change the shader module only
    computePipelineInfo.stage.module = skyShader;

    ComputeEffect skyEffect;
    skyEffect.name = "sky";
    skyEffect.layout = gradientPipelineLayout;
    skyEffect.pipeline = device->createComputePipeline(nullptr, computePipelineInfo);
    skyEffect.data = {
        .data1 = glm::vec4(0.1f, 0.2f, 0.4f, 0.97f), // Sky color + star threshold
        .data2 = glm::vec4(0.0f),
        .data3 = glm::vec4(0.0f),
        .data4 = glm::vec4(0.0f)};

    // Add the 2 background effects into the array
    backgroundEffects.emplace_back(std::move(gradientEffect));
    backgroundEffects.emplace_back(std::move(skyEffect));

    // gradientPipeline = device->createComputePipeline(nullptr, computePipelineInfo);
}

void DemoApp::initDescriptorSets() {
    Application::initDescriptorSets();
}

void DemoApp::updateImGui() {
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    if (ImGui::Begin("background")) {
        ComputeEffect& selected = backgroundEffects[currentBackgroundEffect];
    
        ImGui::Text("Selected effect: %s", selected.name.c_str());
    
        ImGui::SliderInt("Effect Index", &currentBackgroundEffect,0, backgroundEffects.size() - 1);
    
        ImGui::InputFloat4("data1",(float*)& selected.data.data1);
        ImGui::InputFloat4("data2",(float*)& selected.data.data2);
        ImGui::InputFloat4("data3",(float*)& selected.data.data3);
        ImGui::InputFloat4("data4",(float*)& selected.data.data4);
    }
    ImGui::End();
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
        computeQueue.submit2(submitInfo, {});
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
    ComputeEffect& effect = backgroundEffects[currentBackgroundEffect];

    commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, effect.pipeline);
    commandBuffer.bindDescriptorSets(
        vk::PipelineBindPoint::eCompute, gradientPipelineLayout,
        0, *drawImageDescriptorSets[0], {});
    
    ComputePushConstants pushConstants{
        .data1 = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f),
        .data2 = glm::vec4(0.0f, 0.0f, 1.0f, 1.0f)};
    commandBuffer.pushConstants<decltype(effect.data)>(
        gradientPipelineLayout, vk::ShaderStageFlagBits::eCompute,
        0, effect.data);

    commandBuffer.dispatch(
        std::ceil(drawImage.imageExtent.width / 16.0f),
        std::ceil(drawImage.imageExtent.height / 16.0f),
        1);
}