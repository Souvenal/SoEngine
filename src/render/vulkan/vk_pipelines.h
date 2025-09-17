#pragma once

#include "common/vk_common.h"

#include <string>

namespace vkutil
{

/**
 * @brief Creates a Vulkan shader module from a file.
 */
[[nodiscard]]
vk::raii::ShaderModule loadShaderModule(const vk::raii::Device& device,
                                        const std::string& filePath);

}   // namespace vkutil

/**
 * @brief Builder class for creating Vulkan graphics pipelines.
 */
class PipelineBuilder {
public:
    PipelineBuilder();
    ~PipelineBuilder() = default;

    // Non-copyable, moveable
    PipelineBuilder(const PipelineBuilder&) = delete;
    PipelineBuilder& operator=(const PipelineBuilder&) = delete;
    PipelineBuilder(PipelineBuilder&&) = default;
    PipelineBuilder& operator=(PipelineBuilder&&) = default;

    void clear();

    [[nodiscard]] vk::raii::Pipeline buildPipeline(
            const vk::raii::Device& device) const;

    // Shader configuration
    void setShaders(
            const vk::raii::ShaderModule& vertexShader, 
            const vk::raii::ShaderModule& fragmentShader);
    
    // Input assembly
    void setInputTopology(vk::PrimitiveTopology topology);
    
    // Rasterization
    void setPolygonMode(vk::PolygonMode mode);
    void setCullMode(vk::CullModeFlags cullMode, vk::FrontFace frontFace);
    
    // Multisampling
    void setMultisamplingNone();
    
    // Blending
    void disableBlending();
    void enableBlendingAdditive();
    void enableBlendingAlphablend();
    
    // Render targets
    void setColorAttachmentFormat(vk::Format format);
    void setDepthAttachmentFormat(vk::Format format);
    
    // Depth testing
    void disableDepthtest();
    /**
     * @param depthWriteEnable Whether to enable depth writes
     * @param op Use vk::CompareOp::eGreaterOrEqual for reverse-Z
     */
    void enableDepthtest(bool depthWriteEnable, vk::CompareOp op);
    
    // Pipeline layout
    void setPipelineLayout(vk::PipelineLayout layout);

private:
    std::vector<vk::PipelineShaderStageCreateInfo> shaderStages;
    
    vk::PipelineInputAssemblyStateCreateInfo inputAssembly;
    vk::PipelineRasterizationStateCreateInfo rasterizer;
    vk::PipelineColorBlendAttachmentState colorBlendAttachment;
    vk::PipelineMultisampleStateCreateInfo multisampling;
    vk::PipelineDepthStencilStateCreateInfo depthStencil;
    vk::PipelineRenderingCreateInfo renderingInfo;
    
    vk::PipelineLayout pipelineLayout;
    vk::Format colorAttachmentFormat { vk::Format::eUndefined };
    vk::Format depthAttachmentFormat { vk::Format::eUndefined };
};