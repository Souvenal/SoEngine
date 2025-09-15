#pragma once

#include "application.h"

struct ComputeAppInfo {
    const uint32_t particleCount;

    ComputeAppInfo(uint32_t particleCount = 8192)
        : particleCount(particleCount) { }
};

class ComputeApp : public Application {
public:
    ComputeApp(
            const std::filesystem::path& appDir,
            const Window* window,
            const AppInfo& appInfo,
            const ComputeAppInfo& computeInfo);
    ~ComputeApp();

    void onInit() override;
    void onUpdate(double deltaTime) override;
    // void onInputEvent(const InputEvent& event) override;
    void onShutdown() override;

protected:
    ComputeAppInfo computeAppInfo;

    struct UniformBufferObject {
        float deltaTime = 1.0f;
    };

    struct Particle {
        glm::vec2 position;
        glm::vec2 velocity;
        glm::vec4 color;

        static vk::VertexInputBindingDescription getBindingDescription();
        static std::array<vk::VertexInputAttributeDescription, 2> getAttributeDescriptions();
    };

    vk::raii::DescriptorSetLayout       computeDescriptorSetLayout { nullptr };
    vk::raii::PipelineLayout            computePipelineLayout { nullptr };
    vk::raii::Pipeline                  computePipeline { nullptr };

    vk::raii::PipelineLayout            graphicsPipelineLayout { nullptr };
    vk::raii::Pipeline                  graphicsPipeline { nullptr };

    std::vector<AllocatedBuffer>       uniformBuffers;
    std::vector<AllocatedBuffer>       shaderStorageBuffers;

    std::vector<vk::raii::DescriptorSet> computeDescriptorSets;

    vk::raii::CommandBuffers    graphicsCommandBuffers { nullptr };
    vk::raii::CommandBuffers    computeCommandBuffers { nullptr };

protected:
    void initCommandBuffers() override;
    
    void initDescriptorAllocator() override;

    void initDescriptorSetLayouts() override;

    void initPipelines() override;

    void initComputePipeline();

    void initGraphicsPipeline();

    void initUniformBuffers();

    void initShaderStorageBuffers();

    void initDescriptorSets() override;

    void updateUniformBuffer(double deltaTime);

    void drawFrame() override;

    void recordComputeCommandBuffer();

    void drawScene(const vk::raii::CommandBuffer& commandBuffer, uint32_t imageIndex) override;
};