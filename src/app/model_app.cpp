#include "model_app.h"

#include "render/vulkan/vk_pipelines.h"
#include "render/vulkan/vk_images.h"
#include "render/vulkan/vk_utils.h"
#include "common/ktx_error.h"

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include <tiny_gltf.h>

vk::VertexInputBindingDescription ModelApp::Vertex::getBindingDescription() {
    vk::VertexInputBindingDescription bindingDescription {
        .binding = 0, // index of the binding in the array of bindings
        .stride = sizeof(Vertex), // the number of bytes from one entry to the next
        .inputRate = vk::VertexInputRate::eVertex
    };

    return bindingDescription;
}

std::array<vk::VertexInputAttributeDescription, 3> ModelApp::Vertex::getAttributeDescriptions() {
    return {
        vk::VertexInputAttributeDescription( 0, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, pos) ),
        vk::VertexInputAttributeDescription( 1, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, color) ),
        vk::VertexInputAttributeDescription( 2, 0, vk::Format::eR32G32Sfloat, offsetof(Vertex, texCoord) )
    };
}

bool ModelApp::Vertex::operator==(const Vertex& other) const {
    return
        pos == other.pos &&
        color == other.color &&
        texCoord == other.texCoord;
}

ModelApp::ModelApp(
        const std::filesystem::path& appDir,
        const AppInfo& appInfo,
        const ModelAppInfo& modelAppInfo)
    : Application(appDir, appInfo), modelAppInfo(modelAppInfo) { }

ModelApp::~ModelApp() { }

void ModelApp::onInit(const Window* window) {
    // Application::onInit(window);
    LOG_CORE_INFO("===Model application initialization start===");

    initInstance("Model Application");
    initDebugMessenger();
    surface = window->createSurface(*instance);        // influence physical device selection
    auto framebufferSize = window->getFramebufferSize();
    swapchainExtent.width = framebufferSize.width;
    swapchainExtent.height = framebufferSize.height;
    windowExtent = window->getWindowSize();

    selectPhysicalDevice();
    modelAppInfo.msaaSamples = vkutil::getMaxUsableSampleCount(*physicalDevice);

    appInfo.checkFeatureSupport(*instance, *physicalDevice);

    initLogicalDevice();
    
    initMemoryAllocator();
    initSwapchain();

    initCommandPools();
    initCommandBuffers();

    if (!appInfo.dynamicRenderingSupported) {
        initRenderPass();
    }
    initDescriptorAllocator();
    initDescriptorSetLayouts();
    initGraphicsPipeline();

    initColorResources();
    initDepthResources();
    if (!appInfo.dynamicRenderingSupported) {
        initFramebuffers();
    }

    // loadModel("amiya-arknights");
    loadModel("viking_room");
    setupModelObjects();
    initVertexBuffer();
    initIndexBuffer();
    initTextureImage();
    initTextureImageView();
    initTextureSampler();
    initUniformBuffers();
    initDescriptorSets();

    initSyncObjects();

    // Print feature support summary
    // appInfo.printFeatureSupportSummary();

    buildCapabilitiesSummary();
    logCapabilitiesSummary();

    LOG_CORE_INFO("===Model application initialization done===");
}

void ModelApp::onUpdate(double deltaTime) {
    updateUniformBuffer(deltaTime);
}

void ModelApp::onRender() {
    drawFrame();
}

void ModelApp::onShutdown() {
    Application::onShutdown();
}

void ModelApp::recreateSwapchain() {
    ASSERT(device, "Logical device must be created before recreating swap chain");
    device->waitIdle();
    cleanupSwapchain();

    initSwapchain();
    initColorResources();
    initDepthResources();
    // createFramebuffers();
}

void ModelApp::initCommandBuffers() {
    graphicsCommandBuffers.clear();

    vk::CommandBufferAllocateInfo allocInfo {
        .commandPool = *graphicsCommandPool,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = appInfo.maxFramesInFlight
    };

    graphicsCommandBuffers = vk::raii::CommandBuffers(*device, allocInfo);

    LOG_CORE_DEBUG("Graphics command buffers are successfully initialized");
}

void ModelApp::initRenderPass() {
    vk::AttachmentDescription colorAttachment {
        .format = swapchainImageFormat,
        .samples = modelAppInfo.msaaSamples,
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
        .format = vkutil::findDepthFormat(*physicalDevice),
        .samples = modelAppInfo.msaaSamples,
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
        .format = swapchainImageFormat,
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
        .pDependencies = &dependency};

    renderPass = vk::raii::RenderPass(*device, renderPassInfo);

    LOG_CORE_DEBUG("Render pass is successfully initialized");
}

void ModelApp::initDescriptorAllocator() {
    std::vector<DescriptorAllocator::PoolSizeRatio> sizes = {
        { vk::DescriptorType::eUniformBuffer, 1.0f },
        { vk::DescriptorType::eCombinedImageSampler, 1.0f }};
    globalDescriptorAllocator = DescriptorAllocator(
        *device,
        modelAppInfo.maxObjects * appInfo.maxFramesInFlight,
        sizes);

    LOG_CORE_DEBUG("Descriptor allocator is successfully initialized");
}

void ModelApp::initDescriptorSetLayouts() {
    DescriptorLayoutBuilder builder;
    builder.addBinding(0, vk::DescriptorType::eUniformBuffer, vk::ShaderStageFlagBits::eVertex);
    builder.addBinding(1, vk::DescriptorType::eCombinedImageSampler, vk::ShaderStageFlagBits::eFragment);
    drawImageDescriptorSetLayout = builder.build(*device);

    LOG_CORE_DEBUG("Draw image descriptor set layout is successfully initialized");
}

void ModelApp::initGraphicsPipeline() {
    auto shadersDir = appDir / "shaders";
    std::string shaderPath = (shadersDir / "shader.spv").string();
    const auto shaderModule = vkutil::loadShaderModule(*device, shaderPath);

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
        .rasterizationSamples = modelAppInfo.msaaSamples,
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
        .pSetLayouts = &*drawImageDescriptorSetLayout,
        .pushConstantRangeCount = 0, // Optional
        .pPushConstantRanges = nullptr // Optional
    };
    graphicsPipelineLayout = vk::raii::PipelineLayout(*device, pipelineLayoutInfo);

    vk::PipelineRenderingCreateInfo pipelineRenderingCreateInfo {
        .colorAttachmentCount = 1,
        .pColorAttachmentFormats = &swapchainImageFormat,
        .depthAttachmentFormat = vkutil::findDepthFormat(*physicalDevice),
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
        .layout = graphicsPipelineLayout,
        // .renderPass = renderPass,
        .renderPass = nullptr,  // Dynamic rendering
        .basePipelineHandle = VK_NULL_HANDLE, // Optional
        .basePipelineIndex = -1 // Optional
    };

    // Dynamic rendering
    if (appInfo.dynamicRenderingSupported) {
        LOG_CORE_DEBUG("Configuring pipeline for dynamic rendering");
        pipelineInfo.pNext = &pipelineRenderingCreateInfo,
        pipelineInfo.renderPass = nullptr;
    } else {
        LOG_CORE_DEBUG("Configuring pipeline for traditional render pass");
        pipelineInfo.pNext = nullptr;
        pipelineInfo.renderPass = *renderPass;
        pipelineInfo.subpass = 0;
    }

    graphicsPipeline = device->createGraphicsPipeline(nullptr, pipelineInfo);

    LOG_CORE_DEBUG("Graphics pipeline is successfully initialized");
}

void ModelApp::initColorResources() {
    vk::Format colorFormat = swapchainImageFormat;
    colorImage = AllocatedImage(*device, memoryAllocator.allocator,
        {swapchainExtent.width, swapchainExtent.height, 1},
        1, modelAppInfo.msaaSamples,
        colorFormat, vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eTransientAttachment | vk::ImageUsageFlagBits::eColorAttachment,
        vk::ImageAspectFlagBits::eColor,
        MemoryType::DeviceLocal);

    LOG_CORE_DEBUG("Color resources are successfully initialized");
}

void ModelApp::initDepthResources() {
    vk::Format depthFormat = vkutil::findDepthFormat(*physicalDevice);
    depthImage = AllocatedImage(*device, memoryAllocator.allocator,
        {swapchainExtent.width, swapchainExtent.height, 1},
        1, modelAppInfo.msaaSamples,
        depthFormat, vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eDepthStencilAttachment,
        vk::ImageAspectFlagBits::eDepth,
        MemoryType::DeviceLocal);

    LOG_CORE_DEBUG("Depth resources are successfully initialized");
}

void ModelApp::initFramebuffers() {
    if (appInfo.dynamicRenderingSupported) {
        // No framebuffers needed with dynamic rendering
        LOG_CORE_INFO("Using dynamic rendering, skipping framebuffer creation.");
        return;
    }

    swapChainFramebuffers.clear();
    for (size_t i = 0; i < swapchainImageViews.size(); ++i) {
        std::array attachments {
            colorImage.imageView,
            depthImage.imageView,
            *swapchainImageViews[i]
        };
    
        vk::FramebufferCreateInfo framebufferInfo {
            .renderPass = *renderPass,
            .attachmentCount = static_cast<uint32_t>(attachments.size()),
            .pAttachments = attachments.data(),
            .width = swapchainExtent.width,
            .height = swapchainExtent.height,
            .layers = 1
        };

        swapChainFramebuffers.emplace_back(*device, framebufferInfo);
    }

    LOG_CORE_DEBUG("Swap chain framebuffers are successfully initialized");
}

void ModelApp::loadModel(const std::string& modelName) {
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
        LOG_CORE_WARN("glTF: {}", warn);
    }
    if (!err.empty()) {
        LOG_CORE_ERROR("glTF: {}", err);
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
    }
    
    LOG_CORE_INFO("[{}] is successfully loaded with vertices size: {}, indices size: {}", modelName, vertices.size(), indices.size());
}

void ModelApp::setupModelObjects() {
    modelObjects.clear();
    modelObjects.resize(modelAppInfo.maxObjects); // Three instances of the model
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

    LOG_CORE_DEBUG("Model objects are successfully set up");
}

void ModelApp::initVertexBuffer() {
    const vk::DeviceSize bufferSize = sizeof(Vertex) * vertices.size();

    AllocatedBuffer stagingBuffer {
        memoryAllocator.allocator, queueFamilyIndices, bufferSize,
        vk::BufferUsageFlagBits::eTransferSrc,
        MemoryType::HostVisible
    };

    stagingBuffer.write(vertices.data(), vertices.size());

    vertexBuffer = AllocatedBuffer{
        memoryAllocator.allocator, queueFamilyIndices, bufferSize,
        vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer,
        MemoryType::DeviceLocal
    };

    const auto& cmd = vkutil::beginSingleTimeCommands(
        *device, transferCommandPool);
    vkutil::copyAllocatedBuffer(cmd, stagingBuffer, vertexBuffer, bufferSize);
    vkutil::endSingleTimeCommands(cmd, transferQueue);

    LOG_CORE_DEBUG("Vertex buffer is successfully initialized");
}

void ModelApp::initIndexBuffer() {
    const vk::DeviceSize bufferSize = sizeof(indices[0]) * indices.size();

    AllocatedBuffer stagingBuffer{
        memoryAllocator.allocator, queueFamilyIndices, bufferSize,
        vk::BufferUsageFlagBits::eTransferSrc, MemoryType::HostVisible
    };

    stagingBuffer.write(indices.data(), indices.size());

    indexBuffer = AllocatedBuffer{
        memoryAllocator.allocator, queueFamilyIndices, bufferSize,
        vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer,
        MemoryType::DeviceLocal
    };

    const auto& cmd = vkutil::beginSingleTimeCommands(
        *device, transferCommandPool);
    vkutil::copyAllocatedBuffer(cmd, stagingBuffer, indexBuffer, bufferSize);
    vkutil::endSingleTimeCommands(cmd, transferQueue);

    LOG_CORE_DEBUG("Index buffer is successfully initialized");
}

void ModelApp::initTextureImage() {
    // Load KTX2 texture
    ktxVulkanDeviceInfo* vdi = ktxVulkanDeviceInfo_Create(
        **physicalDevice, **device,
        *transferQueue, *transferCommandPool, nullptr);

    ktxTexture2* kTexture;
    KTX_error_code ktxResult = ktxTexture2_CreateFromNamedFile(
        (appDir / modelAppInfo.texturePath).c_str(),
        KTX_TEXTURE_CREATE_NO_FLAGS,
        &kTexture);
    if (ktxResult != KTX_SUCCESS) {
        LOG_CORE_ERROR("Failed to load texture image from file: {}", (appDir / modelAppInfo.texturePath).string());
        handleKtxError(ktxResult);
    }
    mipLevels = kTexture->numLevels;
    kTexture->vkFormat = VK_FORMAT_R8G8B8A8_SRGB;

    ktxResult = ktxTexture2_VkUploadEx(kTexture, vdi, &texture,
        VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    if (ktxResult != KTX_SUCCESS) {
        LOG_CORE_ERROR("Failed to upload texture image to GPU");
        handleKtxError(ktxResult);
    }
    ktxTexture_Destroy(ktxTexture(kTexture));
    ktxVulkanDeviceInfo_Destroy(vdi);

    mainDeletionQueue.pushFunction([=, this]() {
        ktxVulkanTexture_Destruct(&texture, **device, nullptr);
    });

    LOG_CORE_DEBUG("Texture image is successfully initialized");
}

void ModelApp::initTextureImageView() {
    textureImageView = vkutil::createImageView(*device, texture.image,
        static_cast<vk::Format>(texture.imageFormat), vk::ImageAspectFlagBits::eColor, mipLevels);

    LOG_CORE_DEBUG("Texture image view is successfully initialized");
}

void ModelApp::initTextureSampler() {
    auto properties = physicalDevice->getProperties();
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
    textureSampler = vk::raii::Sampler(*device, samplerInfo);

    LOG_CORE_DEBUG("Texture sampler is successfully initialized");
}

void ModelApp::initUniformBuffers() {
    // For each model object
    for (auto& object : modelObjects) {
        object.uniformBuffers.clear();

        for (size_t i = 0; i < appInfo.maxFramesInFlight; ++i) {
            constexpr vk::DeviceSize bufferSize = sizeof(UniformBufferObject);
            AllocatedBuffer buffer{
                memoryAllocator.allocator, queueFamilyIndices, bufferSize,
                vk::BufferUsageFlagBits::eUniformBuffer,
                MemoryType::HostVisible
            };
            object.uniformBuffers.emplace_back(std::move(buffer));
        }
    }

    LOG_CORE_DEBUG("Model uniform buffers are successfully initialized");
}

void ModelApp::initDescriptorSets() {
    // For each model object
    for (auto& object : modelObjects) {
        object.descriptorSets.clear();
        object.descriptorSets = globalDescriptorAllocator.allocate(
            *device, drawImageDescriptorSetLayout, appInfo.maxFramesInFlight);

        for (size_t i = 0; i < appInfo.maxFramesInFlight; ++i) {
            vk::DescriptorBufferInfo bufferInfo {
                .buffer = object.uniformBuffers[i].buffer,
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
            device->updateDescriptorSets(descriptorWrites, {});
        }
    }

    LOG_CORE_DEBUG("Graphics descriptor sets are successfully initialized");
}

void ModelApp::buildCapabilitiesSummary() {
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

void ModelApp::logCapabilitiesSummary() const {
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

void ModelApp::updateUniformBuffer(double deltaTime) {
    static auto startTime = std::chrono::high_resolution_clock::now();

    const auto currentTime = std::chrono::high_resolution_clock::now();
    const float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

    // glm::vec3 objCenter {(minX + maxX) / 2.0f, (minY + maxY) / 2.0f, (minZ + maxZ) / 2.0f };
    float xLen { maxX - minX }, yLen { maxY - minY }, zLen { maxZ - minZ };

    glm::vec3 eye { 0.0f, yLen * 2.5, zLen * 2.5};
    glm::vec3 up {0, 1, 0};
    glm::mat4 view = glm::lookAt(eye, glm::vec3(0.0f), up);
    glm::mat4 proj = glm::perspective(glm::radians(45.0f),
            static_cast<float>(swapchainExtent.width) / static_cast<float>(swapchainExtent.height),
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

        // memcpy(object.uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
        object.uniformBuffers[currentFrame].write(&ubo, 1);
    }
}

void ModelApp::drawFrame() {
    // Acquire an image from the swap chain
    auto [result, imageIndex] = swapchain.acquireNextImage(
        UINT64_MAX, nullptr, inFlightFences[currentFrame]);
    while (vk::Result::eTimeout == device->waitForFences(
        *inFlightFences[currentFrame], vk::True, std::numeric_limits<uint64_t>::max()))
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

    uint64_t graphicsWaitValue = timelineValue;
    uint64_t graphicsSignalValue = ++timelineValue;

    {
        // Record the command buffer

        // drawBackground(graphicsCommandBuffer, imageIndex);
        recordGraphicsCommandBuffer(imageIndex);

        vk::TimelineSemaphoreSubmitInfo timelineSubmitInfo {
            .waitSemaphoreValueCount = 1,
            .pWaitSemaphoreValues = &graphicsWaitValue,
            .signalSemaphoreValueCount = 1,
            .pSignalSemaphoreValues = &graphicsSignalValue,
        };
        
        // This is for model rendering
        // vk::PipelineStageFlags waitDestinationStageMask {
        //     vk::PipelineStageFlagBits::eColorAttachmentOutput
        // };

        // This is for compute particle rendering
        vk::PipelineStageFlags waitStages[] = {
            vk::PipelineStageFlagBits::eVertexInput
        };
        const vk::SubmitInfo submitInfo {
            .pNext = &timelineSubmitInfo,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &*semaphore,
            .pWaitDstStageMask = waitStages,
            .commandBufferCount = 1,
            .pCommandBuffers = &*graphicsCommandBuffers[currentFrame],
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &*semaphore,
        };

        graphicsQueue.submit(submitInfo, nullptr);

        vk::SemaphoreWaitInfo waitInfo {
            .semaphoreCount = 1,
            .pSemaphores = &*semaphore,
            .pValues = &graphicsSignalValue
        };
        while (vk::Result::eTimeout == device->waitSemaphores(waitInfo, std::numeric_limits<uint64_t>::max()))
            ;

        // Presentation
        const vk::PresentInfoKHR presentInfo {
            .waitSemaphoreCount = 0,
            .pWaitSemaphores = nullptr,
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

void ModelApp::recordGraphicsCommandBuffer(uint32_t imageIndex) {
    const auto& commandBuffer = graphicsCommandBuffers[currentFrame];

    commandBuffer.reset();
    commandBuffer.begin({.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    vkutil::transitionImageLayout(commandBuffer,
        // drawImage.image,
        swapchainImages[imageIndex],
        vk::ImageLayout::eUndefined,
        vk::ImageLayout::eColorAttachmentOptimal);

    vk::ClearValue clearColor = vk::ClearColorValue{0.0f, 0.0f, 0.0f, 1.0f};
    vk::ClearValue clearDepth = vk::ClearDepthStencilValue(1.0f, 0);
    vk::ClearValue clearResolve = vk::ClearColorValue{0.0f, 0.0f, 0.0f, 1.0f};
    std::array clearValues = { clearColor, clearDepth, clearResolve};

    if (appInfo.dynamicRenderingSupported) {
        // begin dynamic rendering
        // multisampled color image
        vkutil::transitionImageLayout(commandBuffer,
            colorImage.image,
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::eColorAttachmentOptimal);
        // depth image
        vkutil::transitionImageLayout(commandBuffer,
            depthImage.image,
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::eDepthAttachmentOptimal);

        vk::RenderingAttachmentInfo colorAttachmentInfo {
            .imageView = colorImage.imageView,
            .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .resolveMode = vk::ResolveModeFlagBits::eAverage,
            // .resolveImageView = drawImage.imageView,
            .resolveImageView = swapchainImageViews[imageIndex],
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
                .offset = {0, 0}, .extent = swapchainExtent
            },
            .layerCount = 1,
            .colorAttachmentCount = 1,
            .pColorAttachments = &colorAttachmentInfo,
            .pDepthAttachment = &depthAttachmentInfo,
        };

        commandBuffer.beginRendering(renderingInfo);
    } else {
        throw std::runtime_error("Device does not support dynamic rendering.");
        // Use traditional render pass
        // vk::RenderPassBeginInfo renderPassBeginInfo {
        //     .renderPass = *renderPass,
        //     .framebuffer = *swapChainFramebuffers[imageIndex],
        //     .renderArea = {{0, 0}, swapChainExtent},
        //     .clearValueCount = static_cast<uint32_t>(clearValues.size()),
        //     .pClearValues = clearValues.data(),
        // };

        // getCurrentFrame().commandBuffer.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);
    }

    commandBuffer.setViewport(0,
        vk::Viewport{
            0.0f, 0.0f,
            static_cast<float>(swapchainExtent.width),
            static_cast<float>(swapchainExtent.height),
            0.0f, 1.0f
        });
    commandBuffer.setScissor(0, vk::Rect2D{vk::Offset2D{0, 0}, swapchainExtent});
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *graphicsPipeline);
    // getCurrentFrame().commandBuffer.bindVertexBuffers(0, {vertexBuffer.buffer}, {0});
    commandBuffer.bindVertexBuffers(0, {vertexBuffer.buffer}, {0});
    commandBuffer.bindIndexBuffer(indexBuffer.buffer, 0, vk::IndexType::eUint32);
    // Draw each object with its own descriptor set
    for (const auto& object : modelObjects) {
        commandBuffer.bindDescriptorSets(
            vk::PipelineBindPoint::eGraphics,
            *graphicsPipelineLayout, 0,
            {*object.descriptorSets[currentFrame]}, {});
        // Draw the object
        commandBuffer.drawIndexed(static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);
    }

    if (appInfo.dynamicRenderingSupported) {
        commandBuffer.endRendering();
    } else {
        commandBuffer.endRenderPass();
    }

    // vkutil::transitionImageLayout(cmd,
    //     drawImage.image,
    //     vk::ImageLayout::eColorAttachmentOptimal,
    //     vk::ImageLayout::eTransferSrcOptimal);
    // vkutil::transitionImageLayout(cmd,
    //     swapChainImages[imageIndex],
    //     vk::ImageLayout::eUndefined,
    //     vk::ImageLayout::eTransferDstOptimal);
    // auto drawExtent = vk::Extent2D{ drawImage.imageExtent.width, drawImage.imageExtent.height };
    // vkutil::copyImageToImage(cmd,
    //     drawImage.image,
    //     swapChainImages[imageIndex],
    //     drawExtent,
    //     swapChainExtent
    // );

    // vkutil::transitionImageLayout(cmd,
    //     swapChainImages[imageIndex],
    //     vk::ImageLayout::eTransferDstOptimal,
    //     vk::ImageLayout::eColorAttachmentOptimal
    // );

    // drawGui();

    // vkutil::transitionImageLayout(cmd,
    //     swapChainImages[imageIndex],
    //     vk::ImageLayout::eTransferDstOptimal,
    //     vk::ImageLayout::ePresentSrcKHR
    // );

    vkutil::transitionImageLayout(commandBuffer,
        // drawImage.image,
        swapchainImages[imageIndex],
        vk::ImageLayout::eColorAttachmentOptimal,
        vk::ImageLayout::ePresentSrcKHR);
    
    commandBuffer.end();
}
