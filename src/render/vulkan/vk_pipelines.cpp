#include "vk_pipelines.h"

#include "utils/logging.h"

#include <fstream>

namespace {
std::vector<char> readFile(const std::string& filename) {
    std::ifstream file (filename, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file at " + filename);
    }

    const size_t fileSize = static_cast<size_t>(file.tellg());
    std::vector<char> buffer(fileSize);

    file.seekg(0, std::ios::beg);
    file.read(buffer.data(), static_cast<std::streamsize>(fileSize));
    file.close();

    return buffer;
}
}   // namespace

namespace vkutil
{

vk::raii::ShaderModule loadShaderModule(const vk::raii::Device& device,
                                        const std::string& filePath)
{
    auto code = readFile(filePath);

    vk::ShaderModuleCreateInfo createInfo{
        .codeSize = code.size() * sizeof(char),
        .pCode = reinterpret_cast<const uint32_t*>(code.data())
    };

    return device.createShaderModule(createInfo);
}

} // namespace vkutil

PipelineBuilder::PipelineBuilder() {
    clear();
}

void PipelineBuilder::clear() {
    // Clear all of the structs back to defaults
    inputAssembly = {};
    rasterizer = {};
    colorBlendAttachment = {};
    multisampling = {};
    depthStencil = {};
    renderingInfo = {};

    pipelineLayout = nullptr;

    shaderStages.clear();
}

vk::raii::Pipeline PipelineBuilder::buildPipeline(const vk::raii::Device& device) const {
    // Make viewport state from our stored viewport and scissor.
    // At the moment we won't support multiple viewports or scissors
    vk::PipelineViewportStateCreateInfo viewportState{
        .viewportCount = 1,
        .scissorCount = 1};

    // Setup dummy color blending. We aren't using transparent objects yet
    // The blending is just "no blend", but we do write to the color attachment
    vk::PipelineColorBlendStateCreateInfo colorBlending{
        .logicOpEnable = vk::False,
        .logicOp = vk::LogicOp::eCopy,
        .attachmentCount = 1,
        .pAttachments = &colorBlendAttachment};

    // Completely clear VertexInputStateCreateInfo, as we have no need for it
    vk::PipelineVertexInputStateCreateInfo vertexInputInfo{};

    // Dynamic states
    std::vector<vk::DynamicState> dynamicStates = {
        vk::DynamicState::eViewport, 
        vk::DynamicState::eScissor};

    vk::PipelineDynamicStateCreateInfo dynamicInfo{
        .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
        .pDynamicStates = dynamicStates.data()};

    // Build the actual pipeline
    // We now use all of the info structs we have been writing into this one
    // to create the pipeline
    vk::GraphicsPipelineCreateInfo pipelineInfo{
        .pNext = &renderingInfo,  // Connect the renderInfo to the pNext extension mechanism
        .stageCount = static_cast<uint32_t>(shaderStages.size()),
        .pStages = shaderStages.data(),
        .pVertexInputState = &vertexInputInfo,
        .pInputAssemblyState = &inputAssembly,
        .pViewportState = &viewportState,
        .pRasterizationState = &rasterizer,
        .pMultisampleState = &multisampling,
        .pDepthStencilState = &depthStencil,
        .pColorBlendState = &colorBlending,
        .pDynamicState = &dynamicInfo,
        .layout = pipelineLayout};

    return device.createGraphicsPipeline(nullptr, pipelineInfo);
}

void PipelineBuilder::setShaders(const vk::raii::ShaderModule& vertexShader, 
                                const vk::raii::ShaderModule& fragmentShader) {
    shaderStages.clear();

    vk::PipelineShaderStageCreateInfo vertShaderStageInfo{
        .stage = vk::ShaderStageFlagBits::eVertex,
        .module = vertexShader,
        .pName = "vertMain"};
    shaderStages.emplace_back(vertShaderStageInfo);

    vk::PipelineShaderStageCreateInfo fragShaderStageInfo{
        .stage = vk::ShaderStageFlagBits::eFragment,
        .module = fragmentShader,
        .pName = "fragMain"};
    shaderStages.emplace_back(fragShaderStageInfo);
}

void PipelineBuilder::setInputTopology(vk::PrimitiveTopology topology) {
    inputAssembly.topology = topology;
    // I don't know how to use primitive restart yet
    // So for now, we just disable it
    inputAssembly.primitiveRestartEnable = vk::False;
}

void PipelineBuilder::setPolygonMode(vk::PolygonMode mode) {
    rasterizer.polygonMode = mode;
    rasterizer.lineWidth = 1.0f;
}

void PipelineBuilder::setCullMode(vk::CullModeFlags cullMode, vk::FrontFace frontFace) {
    rasterizer.cullMode = cullMode;
    rasterizer.frontFace = frontFace;
}

void PipelineBuilder::setMultisamplingNone() {
    multisampling.sampleShadingEnable = vk::False;
    // Multisampling defaulted to no multisampling (1 sample per pixel)
    multisampling.rasterizationSamples = vk::SampleCountFlagBits::e1;
    multisampling.minSampleShading = 1.0f;
    multisampling.pSampleMask = nullptr;
    // No alpha to coverage either
    multisampling.alphaToCoverageEnable = vk::False;
    multisampling.alphaToOneEnable = vk::False;
}

void PipelineBuilder::disableBlending() {
    // Default write mask
    colorBlendAttachment.colorWriteMask = 
        vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | 
        vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
    // No blending
    colorBlendAttachment.blendEnable = vk::False;
}

void PipelineBuilder::enableBlendingAdditive() {
    colorBlendAttachment.colorWriteMask = 
        vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | 
        vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
    colorBlendAttachment.blendEnable = vk::True;
    colorBlendAttachment.srcColorBlendFactor = vk::BlendFactor::eSrcAlpha;
    colorBlendAttachment.dstColorBlendFactor = vk::BlendFactor::eOne;
    colorBlendAttachment.colorBlendOp = vk::BlendOp::eAdd;
    colorBlendAttachment.srcAlphaBlendFactor = vk::BlendFactor::eOne;
    colorBlendAttachment.dstAlphaBlendFactor = vk::BlendFactor::eZero;
    colorBlendAttachment.alphaBlendOp = vk::BlendOp::eAdd;
}

void PipelineBuilder::enableBlendingAlphablend() {
    colorBlendAttachment.colorWriteMask = 
        vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | 
        vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
    colorBlendAttachment.blendEnable = vk::True;
    colorBlendAttachment.srcColorBlendFactor = vk::BlendFactor::eSrcAlpha;
    colorBlendAttachment.dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
    colorBlendAttachment.colorBlendOp = vk::BlendOp::eAdd;
    colorBlendAttachment.srcAlphaBlendFactor = vk::BlendFactor::eOne;
    colorBlendAttachment.dstAlphaBlendFactor = vk::BlendFactor::eZero;
    colorBlendAttachment.alphaBlendOp = vk::BlendOp::eAdd;
}

void PipelineBuilder::setColorAttachmentFormat(vk::Format format) {
    colorAttachmentFormat = format;
    // Connect the format to the renderingInfo structure
    // Multiple color attachments can be used for deferred rendering
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachmentFormats = &colorAttachmentFormat;
}

void PipelineBuilder::setDepthAttachmentFormat(vk::Format format) {
    renderingInfo.depthAttachmentFormat = format;
}

void PipelineBuilder::disableDepthtest() {
    depthStencil.depthTestEnable = vk::False;
    depthStencil.depthWriteEnable = vk::False;
    depthStencil.depthCompareOp = vk::CompareOp::eNever;
    depthStencil.depthBoundsTestEnable = vk::False;
    depthStencil.stencilTestEnable = vk::False;
    depthStencil.front = {};
    depthStencil.back = {};
    depthStencil.minDepthBounds = 0.0f;
    depthStencil.maxDepthBounds = 1.0f;
}

void PipelineBuilder::enableDepthtest(bool depthWriteEnable, vk::CompareOp op) {
    depthStencil.depthTestEnable = vk::True;
    depthStencil.depthWriteEnable = depthWriteEnable ? vk::True : vk::False;
    depthStencil.depthCompareOp = op;
    depthStencil.depthBoundsTestEnable = vk::False;
    depthStencil.stencilTestEnable = vk::False;
    depthStencil.front = {};
    depthStencil.back = {};
    depthStencil.minDepthBounds = 0.0f;
    depthStencil.maxDepthBounds = 1.0f;
}

void PipelineBuilder::setPipelineLayout(vk::PipelineLayout layout) {
    pipelineLayout = layout;
}