#include "demo_app.h"

#include "render/vulkan/vk_images.h"
#include "render/vulkan/vk_pipelines.h"
#include "render/vulkan/vk_initializers.h"
#include "render/vulkan/vk_utils.h"

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
    initDepthImage();

    initCommandPools();
    initCommandBuffers();

    initDescriptorSetLayouts();

    initPipelines();

    initDescriptorSets();

    initSyncObjects();

    initImGui();

    initAssets();

    buildCapabilitiesSummary();
    logCapabilitiesSummary();

    LOG_CORE_INFO("====Demo application initialization done====");
}

void DemoApp::onShutdown() {
    Application::onShutdown();
}

void DemoApp::initLogicalDevice() {
    vk::StructureChain<
        vk::PhysicalDeviceFeatures2,
        vk::PhysicalDeviceVulkan11Features,
        vk::PhysicalDeviceVulkan13Features,
        vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT,
        vk::PhysicalDeviceTimelineSemaphoreFeaturesKHR,     // Timeline semaphores
        vk::PhysicalDeviceBufferDeviceAddressFeatures    // Buffer device address (vma)
    > featureChain {
        {.features = {
            .sampleRateShading = true,
            .samplerAnisotropy = true,
            .shaderInt64 = true}},
        {.shaderDrawParameters = true},
        {.synchronization2 =  true, .dynamicRendering = true},
        {.extendedDynamicState = true},
        {.timelineSemaphore = true},
        {.bufferDeviceAddress = true}};
    Application::initLogicalDevice(featureChain);
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
    // Compute pipelines
    initBackgroundPipelines();

    // Graphics pipelines
    initMeshPipeline();
}

void DemoApp::initMeshPipeline() {
    auto shadersDir = appDir / "shaders";
    auto shaderModule = vkutil::loadShaderModule(*device, shadersDir / "mesh.spv");

    vk::PushConstantRange bufferRange {
        .stageFlags = vk::ShaderStageFlagBits::eVertex,
        .offset = 0,
        .size = sizeof(GPUDrawPushConstants)};
    vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &bufferRange};
    meshPipelineLayout = device->createPipelineLayout(pipelineLayoutInfo);

    PipelineBuilder pipelineBuilder;
    pipelineBuilder.setPipelineLayout(meshPipelineLayout);
    pipelineBuilder.setShaders(shaderModule, shaderModule);
    // Input assembly
    pipelineBuilder.setInputTopology(vk::PrimitiveTopology::eTriangleList);
    // Rasterizer
    // - Counter clockwise for Y-flip
    pipelineBuilder.setPolygonMode(vk::PolygonMode::eFill);
    pipelineBuilder.setCullMode(vk::CullModeFlagBits::eBack, vk::FrontFace::eCounterClockwise);
    // Multisampling
    pipelineBuilder.setMultisamplingNone();
    // Blending
    pipelineBuilder.disableBlending();
    // Depth testing
    pipelineBuilder.enableDepthtest(true, vk::CompareOp::eGreaterOrEqual); // reverse Z

    // // Vertex input
    // auto bindingDescription = Vertex::getBindingDescription();
    // auto attributeDescriptions = Vertex::getAttributeDescriptions();
    // pipelineBuilder.vertexInputInfo.vertexBindingDescriptionCount = 1;
    // pipelineBuilder.vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
    // pipelineBuilder.vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
    // pipelineBuilder.vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

    pipelineBuilder.setColorAttachmentFormat(drawImage.imageFormat);
    pipelineBuilder.setDepthAttachmentFormat(depthImage.imageFormat);

    meshPipeline = pipelineBuilder.buildPipeline(*device);
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

void DemoApp::initAssets() {
    auto modelsDir = appDir / "assets" / "models";
    auto meshes = vkutil::loadGltfMeshes(*this, modelsDir / "basicmesh.glb");
    if (meshes) {
        testMeshes = std::move(*meshes);
        LOG_CORE_INFO("Loaded {} meshes from basicmesh.glb", testMeshes.size());
    } else {
        LOG_CORE_ERROR(meshes.error());
    }
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
            graphicsCommandBuffer, drawImage.image,
            vk::ImageLayout::eGeneral,
            vk::ImageLayout::eColorAttachmentOptimal);
        vkutil::transitionImageLayout(
            graphicsCommandBuffer, depthImage.image,
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::eDepthAttachmentOptimal);

        drawGeometry(graphicsCommandBuffer);

        vkutil::transitionImageLayout(
            graphicsCommandBuffer, drawImage.image,
            vk::ImageLayout::eColorAttachmentOptimal,
            vk::ImageLayout::eTransferSrcOptimal);
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

void DemoApp::drawGeometry(const vk::raii::CommandBuffer& commandBuffer) {
    auto colorAttachmentInfo = vkinit::colorAttachmentInfo(
        drawImage.imageView, vk::ImageLayout::eColorAttachmentOptimal);
    auto depthAttachmentInfo = vkinit::depthAttachmentInfo(
        depthImage.imageView, vk::ImageLayout::eDepthAttachmentOptimal);
    vk::Extent2D drawExtent = {drawImage.imageExtent.width, drawImage.imageExtent.height};
    auto renderingInfo = vkinit::renderingInfo(drawExtent, colorAttachmentInfo, depthAttachmentInfo);

    commandBuffer.beginRendering(renderingInfo);

    // Set viewport and scissor
    vk::Viewport viewport{
        .x = 0.0f, .y = 0.0f,
        .width = static_cast<float>(drawExtent.width),
        .height = static_cast<float>(drawExtent.height),
        .minDepth = 0.0f,
        .maxDepth = 1.0f};
    commandBuffer.setViewport(0, {viewport});
    vk::Rect2D scissor{
        .offset = {0, 0},
        .extent = drawExtent};
    commandBuffer.setScissor(0, {scissor});

    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, meshPipeline);

    GPUDrawPushConstants pushConstants;
    glm::mat4 view = glm::translate(glm::vec3{ 0,0,-5 });
	// camera projection
	glm::mat4 projection = glm::perspective(
        glm::radians(70.f),
        (float)drawExtent.width / (float)drawExtent.height,
        10000.f, 0.1f);

	// invert the Y direction on projection matrix so that we are more similar
	// to opengl and gltf axis
	projection[1][1] *= -1;
    pushConstants.worldMatrix = projection * view;
    pushConstants.vertexBuffer = testMeshes[2]->meshBuffers.vertexBufferAddress;

    commandBuffer.pushConstants<GPUDrawPushConstants>(
        meshPipelineLayout, vk::ShaderStageFlagBits::eVertex,
        0, pushConstants);
    commandBuffer.bindIndexBuffer(
        testMeshes[2]->meshBuffers.indexBuffer.buffer, 0, vk::IndexType::eUint32);
    commandBuffer.drawIndexed(
        testMeshes[2]->surfaces[0].count, 1,
        testMeshes[2]->surfaces[0].startIndex, 0, 0);

    commandBuffer.endRendering();
}