#include "compute_app.h"

#include "render/vulkan/vk_pipelines.h"
#include "render/vulkan/vk_images.h"
#include "render/vulkan/vk_utils.h"
#include "render/vulkan/vk_initializers.h"

#include <numbers>
#include <random>

vk::VertexInputBindingDescription ComputeApp::Particle::getBindingDescription() {
    return { 0, sizeof(Particle), vk::VertexInputRate::eVertex };
}

std::array<vk::VertexInputAttributeDescription, 2> ComputeApp::Particle::getAttributeDescriptions() {
    return {
        vk::VertexInputAttributeDescription( 0, 0, vk::Format::eR32G32Sfloat, offsetof(Particle, position) ),
        vk::VertexInputAttributeDescription( 1, 0, vk::Format::eR32G32B32A32Sfloat, offsetof(Particle, color) ),
    };
}

ComputeApp::ComputeApp(
        const std::filesystem::path& appDir,
        const Window* window,
        const AppInfo& appInfo,
        const ComputeAppInfo& computeAppInfo):
    Application(appDir, window, appInfo),
    computeAppInfo(computeAppInfo) { }

ComputeApp::~ComputeApp() { }

void ComputeApp::onInit() {
    LOG_CORE_INFO("===Compute application initialization start===");

    initInstance("Compute application");
    // initDebugMessenger();
    initSurface();

    selectPhysicalDevice();

    appInfo.checkFeatureSupport(*instance, *physicalDevice);

    initLogicalDevice();
    initMemoryAllocator();
    initSwapchain();

    initCommandPools();
    initCommandBuffers();

    // if (!appInfo.profileSupported && !appInfo.dynamicRenderingSupported) {
    //     initRenderPass();
    // }
    initDescriptorAllocator();
    initDescriptorSetLayouts();
    initPipelines();
    
    // if (!appInfo.profileSupported && !appInfo.dynamicRenderingSupported) {
    //     initFramebuffers();
    // }

    initUniformBuffers();
    initShaderStorageBuffers();
    initDescriptorSets();

    initSyncObjects();

    initImGui();

    buildCapabilitiesSummary();
    logCapabilitiesSummary();

    LOG_CORE_INFO("===Compute application initialization done===");
}

void ComputeApp::onUpdate(double deltaTime) {
    Application::onUpdate(deltaTime);
    updateUniformBuffer(deltaTime);
}

void ComputeApp::onShutdown() {
    Application::onShutdown();
}

void ComputeApp::initCommandBuffers() {
    vk::CommandBufferAllocateInfo allocInfo {
        .commandPool = *graphicsCommandPool,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = appInfo.maxFramesInFlight
    };
    graphicsCommandBuffers = vk::raii::CommandBuffers(*device, allocInfo);

    vk::CommandBufferAllocateInfo computeAllocInfo {
        .commandPool = *computeCommandPool,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = appInfo.maxFramesInFlight
    };
    computeCommandBuffers = vk::raii::CommandBuffers(*device, computeAllocInfo);

    LOG_CORE_DEBUG("Graphics and compute command buffers are successfully initialized");
}

void ComputeApp::initDescriptorAllocator() {
    uint32_t particleSets = appInfo.maxFramesInFlight;
    uint32_t imguiSets = appInfo.maxFramesInFlight;
    uint32_t maxSets = particleSets + imguiSets;
    std::vector<DescriptorAllocator::PoolSizeRatio> sizes = {
        { vk::DescriptorType::eUniformBuffer, 1.0f },
        { vk::DescriptorType::eStorageBuffer, 2.0f },
        { vk::DescriptorType::eCombinedImageSampler, 1.0f }
    };
    globalDescriptorAllocator = DescriptorAllocator(*device, maxSets, sizes);
}

void ComputeApp::initDescriptorSetLayouts() {
    DescriptorLayoutBuilder builder;
    builder.addBinding(0, vk::DescriptorType::eUniformBuffer, vk::ShaderStageFlagBits::eCompute);
    builder.addBinding(1, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute);
    builder.addBinding(2, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute);
    computeDescriptorSetLayout = builder.build(*device);

    LOG_CORE_DEBUG("Compute descriptor set layout is successfully initialized");
}

void ComputeApp::initPipelines() {
    initComputePipeline();
    initGraphicsPipeline();
    LOG_CORE_DEBUG("Pipelines are successfully initialized");
}

void ComputeApp::initComputePipeline() {
    const auto shadersDir = appDir / "shaders";
    const auto computeShaderPath = shadersDir / "compute.spv";
    auto computeShaderModule = vkutil::loadShaderModule(*device, computeShaderPath.string());

    vk::PipelineLayoutCreateInfo layoutInfo {
        .setLayoutCount = 1,
        .pSetLayouts = &*computeDescriptorSetLayout
    };
    computePipelineLayout = device->createPipelineLayout(layoutInfo);

    vk::ComputePipelineCreateInfo pipelineInfo {
        .stage = vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eCompute,
            .module = computeShaderModule,
            .pName = "compMain"
        },
        .layout = computePipelineLayout
    };

    // computePipeline = device->createComputePipeline(nullptr, pipelineInfo);
    computePipeline = vk::raii::Pipeline(*device, nullptr, pipelineInfo);

    LOG_CORE_DEBUG("Compute pipeline is successfully initialized");
}

void ComputeApp::initGraphicsPipeline() {
    auto shadersDir = appDir / "shaders";
    auto shaderPath = (shadersDir / "compute.spv");
    auto shaderModule = vkutil::loadShaderModule(*device, shaderPath.string());

    vk::PipelineShaderStageCreateInfo shaderStageInfo {
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
        shaderStageInfo, fragShaderStageInfo
    };

    // Vertex input
    auto bindingDescription = Particle::getBindingDescription();
    auto attributeDescriptions = Particle::getAttributeDescriptions();
    vk::PipelineVertexInputStateCreateInfo vertexInputInfo {
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = &bindingDescription,
        .vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size()),
        .pVertexAttributeDescriptions = attributeDescriptions.data()
    };

    vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
        .topology = vk::PrimitiveTopology::ePointList,
        .primitiveRestartEnable = vk::False
    };
    vk::PipelineViewportStateCreateInfo viewportState{
        .viewportCount = 1,
        .scissorCount = 1
    };

    vk::PipelineRasterizationStateCreateInfo rasterizer{
        .depthClampEnable = vk::False,
        .rasterizerDiscardEnable = vk::False,
        .polygonMode = vk::PolygonMode::eFill,
        .cullMode = vk::CullModeFlagBits::eBack,
        .frontFace = vk::FrontFace::eCounterClockwise,
        .depthBiasEnable = vk::False,
        .lineWidth = 1.0f
    };

    vk::PipelineMultisampleStateCreateInfo multisampling{
        .rasterizationSamples = vk::SampleCountFlagBits::e1,
        .sampleShadingEnable = vk::False
    };

    vk::PipelineColorBlendAttachmentState colorBlendAttachment{
        .blendEnable = vk::True,
        .srcColorBlendFactor = vk::BlendFactor::eSrcAlpha,
        .dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
        .colorBlendOp = vk::BlendOp::eAdd,
        .srcAlphaBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
        .dstAlphaBlendFactor = vk::BlendFactor::eZero,
        .alphaBlendOp = vk::BlendOp::eAdd,
        .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
    };
    vk::PipelineColorBlendStateCreateInfo colorBlending{
        .logicOpEnable = vk::False,
        .logicOp = vk::LogicOp::eCopy,
        .attachmentCount = 1,
        .pAttachments = &colorBlendAttachment
    };

    std::vector dynamicStates = {
        vk::DynamicState::eViewport,
        vk::DynamicState::eScissor
    };
    vk::PipelineDynamicStateCreateInfo dynamicState{
        .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
        .pDynamicStates = dynamicStates.data()
    };

    vk::PipelineLayoutCreateInfo layoutInfo {};
    graphicsPipelineLayout = device->createPipelineLayout(layoutInfo);

    vk::PipelineRenderingCreateInfo pipelineRenderingCreateInfo {
        .colorAttachmentCount = 1,
        .pColorAttachmentFormats = &swapchainImageFormat,
    };
    vk::GraphicsPipelineCreateInfo pipelineInfo {
        .pNext = &pipelineRenderingCreateInfo,
        .stageCount = 2,
        .pStages = shaderStages,
        .pVertexInputState = &vertexInputInfo,
        .pInputAssemblyState = &inputAssembly,
        .pViewportState = &viewportState,
        .pRasterizationState = &rasterizer,
        .pMultisampleState = &multisampling,
        .pColorBlendState = &colorBlending,
        .pDynamicState = &dynamicState,
        .layout = graphicsPipelineLayout
    };

    // graphicsPipeline = device->createGraphicsPipeline(nullptr, pipelineInfo);
    graphicsPipeline = vk::raii::Pipeline(*device, nullptr, pipelineInfo);

    LOG_CORE_DEBUG("Graphics pipeline is successfully initialized");
}

void ComputeApp::initShaderStorageBuffers() {
    std::random_device rd {};
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    std::vector<Particle> particles(computeAppInfo.particleCount);
    for (auto& particle : particles) {
        float r = 0.25f * sqrtf(dis(gen));
        float theta = dis(gen) * 2.0f * std::numbers::pi_v<float>;
        float x = r * cosf(theta) * windowExtent.height / windowExtent.width;
        float y = r * sinf(theta);
        particle.position = glm::vec2(x, y);
        particle.velocity = normalize(glm::vec2(x,y)) * 0.00025f;
        particle.color = glm::vec4(dis(gen), dis(gen), dis(gen), 1.0f);
        // particle.color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
    }

    vk::DeviceSize bufferSize = sizeof(Particle) * particles.size();
    LOG_CORE_DEBUG("Particle buffer size: {}", bufferSize);

    AllocatedBuffer stagingBuffer{
        memoryAllocator.allocator, queueFamilyIndices, bufferSize,
        vk::BufferUsageFlagBits::eTransferSrc, MemoryType::HostVisible
    };
    stagingBuffer.write(particles.data(), particles.size());

    shaderStorageBuffers.clear();
    // Copy initial particle data to all storage buffers
    for (size_t i = 0; i < appInfo.maxFramesInFlight; ++i) {
        AllocatedBuffer buffer{
            memoryAllocator.allocator, queueFamilyIndices, bufferSize,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst,
            MemoryType::DeviceLocal
        };
        auto cmd = vkutil::beginSingleTimeCommands(*device, transferCommandPool);
        vkutil::copyAllocatedBuffer(cmd, stagingBuffer, buffer, bufferSize);
        vkutil::endSingleTimeCommands(cmd, transferQueue);

        shaderStorageBuffers.emplace_back(std::move(buffer));
    }

    LOG_CORE_DEBUG("Shader storage buffers are successfully initialized");
}

void ComputeApp::initUniformBuffers() {
    uniformBuffers.clear();

    for (size_t i = 0; i < appInfo.maxFramesInFlight; ++i) {
        constexpr vk::DeviceSize bufferSize = sizeof(UniformBufferObject);
        AllocatedBuffer buffer {
            memoryAllocator.allocator, queueFamilyIndices, bufferSize,
            vk::BufferUsageFlagBits::eUniformBuffer,
            MemoryType::HostVisible
        };
        uniformBuffers.emplace_back(std::move(buffer));
    }

    LOG_CORE_DEBUG("Particle uniform buffers are successfully initialized");
}

void ComputeApp::initDescriptorSets() {
    computeDescriptorSets.clear();
    computeDescriptorSets = globalDescriptorAllocator.allocate(*device, computeDescriptorSetLayout, appInfo.maxFramesInFlight);
    for (size_t i = 0; i < appInfo.maxFramesInFlight; ++i) {
        vk::DescriptorBufferInfo bufferInfo(
            uniformBuffers[i].buffer, 0, sizeof(UniformBufferObject));
        vk::DescriptorBufferInfo storageBufferInfoLastFrame(
            shaderStorageBuffers[(i - 1) % appInfo.maxFramesInFlight].buffer, 0,
            sizeof(Particle) * computeAppInfo.particleCount);
        vk::DescriptorBufferInfo storageBufferInfoCurrentFrame(
            shaderStorageBuffers[i].buffer, 0,
            sizeof(Particle) * computeAppInfo.particleCount);

        std::array descriptorWrites{
            vk::WriteDescriptorSet {
                .dstSet = *computeDescriptorSets[i],
                .dstBinding = 0,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eUniformBuffer,
                .pBufferInfo = &bufferInfo },
            vk::WriteDescriptorSet{
                .dstSet = *computeDescriptorSets[i],
                .dstBinding = 1,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eStorageBuffer,
                .pBufferInfo = &storageBufferInfoLastFrame
            },
            vk::WriteDescriptorSet{
                .dstSet = *computeDescriptorSets[i],
                .dstBinding = 2,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eStorageBuffer,
                .pBufferInfo = &storageBufferInfoCurrentFrame
            },
        };
        device->updateDescriptorSets(descriptorWrites, {});
    }

    LOG_CORE_DEBUG("Compute descriptor sets are successfully initialized");
}

void ComputeApp::updateUniformBuffer(double deltaTime) {
    UniformBufferObject ubo{};
    ubo.deltaTime = static_cast<float>(deltaTime) * 2000.0f;
    // LOG_CORE_DEBUG("{}", ubo.deltaTime * 1000.f);

    uniformBuffers[currentFrame].write(&ubo, 1);
}

void ComputeApp::drawFrame() {
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
    uint64_t graphicsWaitValue = computeSignalValue;
    uint64_t graphicsSignalValue = ++timelineValue;

    // Compute Task
    {
        recordComputeCommandBuffer();
        vk::TimelineSemaphoreSubmitInfo timelineSubmitInfo {
            .waitSemaphoreValueCount = 1,
            .pWaitSemaphoreValues = &computeWaitValue,
            .signalSemaphoreValueCount = 1,
            .pSignalSemaphoreValues = &computeSignalValue,
        };

        vk::PipelineStageFlags waitStages[] = {
            vk::PipelineStageFlagBits::eComputeShader
        };
        const vk::SubmitInfo computeSubmitInfo {
            .pNext = &timelineSubmitInfo,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &*semaphore,
            .pWaitDstStageMask = waitStages,
            .commandBufferCount = 1,
            .pCommandBuffers = &*computeCommandBuffers[currentFrame],
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &*semaphore,
        };

        computeQueue.submit(computeSubmitInfo, nullptr);
    }
    // Graphics Task
    {
        const auto& commandBuffer = graphicsCommandBuffers[currentFrame];
        commandBuffer.reset();
        commandBuffer.begin({.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    
        vkutil::transitionImageLayout(
            commandBuffer, swapchainImages[imageIndex],
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::eColorAttachmentOptimal);

        drawScene(commandBuffer, imageIndex);

        drawImGui(commandBuffer, imageIndex);

        vkutil::transitionImageLayout(commandBuffer,
            swapchainImages[imageIndex],
            vk::ImageLayout::eColorAttachmentOptimal,
            vk::ImageLayout::ePresentSrcKHR
        );

        commandBuffer.end();

        auto commandBufferInfo = vkinit::commandBufferSubmitInfo(*graphicsCommandBuffers[currentFrame]);
        auto waitSubmitInfo = vkinit::semaphoreSubmitInfo(
            vk::PipelineStageFlagBits2::eVertexInput, *semaphore, graphicsWaitValue);
        auto signalSubmitInfo = vkinit::semaphoreSubmitInfo(
            vk::PipelineStageFlagBits2::eAllCommands, *semaphore, graphicsSignalValue);
        auto submitInfo = vkinit::submitInfo(
            commandBufferInfo, waitSubmitInfo, signalSubmitInfo);

        graphicsQueue.submit2(submitInfo, {});

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
    }
    currentFrame = (currentFrame + 1) % appInfo.maxFramesInFlight;
}

void ComputeApp::recordComputeCommandBuffer() {
    const auto& commandBuffer = computeCommandBuffers[currentFrame];
    commandBuffer.reset();
    commandBuffer.begin({});
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *computePipeline);
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *computePipelineLayout, 0, {computeDescriptorSets[currentFrame]}, {});
    commandBuffer.dispatch( computeAppInfo.particleCount / 256, 1, 1 );
    commandBuffer.end();
}

void ComputeApp::drawScene(const vk::raii::CommandBuffer& commandBuffer, uint32_t imageIndex) {
    vk::ClearValue clearColor = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f);
    auto colorAttachment = vkinit::colorAttachmentInfo(
        swapchainImageViews[imageIndex], vk::ImageLayout::eColorAttachmentOptimal, clearColor);
    auto renderingInfo = vkinit::renderingInfo(swapchainExtent, colorAttachment);
    commandBuffer.beginRendering(renderingInfo);

    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *graphicsPipeline);
    commandBuffer.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapchainExtent.width), static_cast<float>(swapchainExtent.height), 0.0f, 1.0f));
    commandBuffer.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapchainExtent));
    commandBuffer.bindVertexBuffers(0, { shaderStorageBuffers[currentFrame].buffer }, {0});
    commandBuffer.draw( computeAppInfo.particleCount, 1, 0, 0 );

    commandBuffer.endRendering();
}