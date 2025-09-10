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
            const AppInfo& appInfo,
            const ComputeAppInfo& computeInfo);
    ~ComputeApp();

    void onInit(const Window* window) override;
    void onUpdate(double deltaTime) override;
    void onRender() override;
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

    std::vector<AllocatedBuffer>       uniformBuffers;
    std::vector<AllocatedBuffer>       shaderStorageBuffers;

    std::vector<vk::raii::DescriptorSet> computeDescriptorSets;

    std::vector<vk::raii::CommandBuffer>    graphicsCommandBuffers;
    std::vector<vk::raii::CommandBuffer>    computeCommandBuffers;

protected:
    void initCommandBuffers() override;
    
    void initDescriptorAllocator() override;

    void initComputeDescriptorSetLayout();

    void initComputePipeline();

    void initGraphicsPipeline() override;

    void initUniformBuffers();

    void initShaderStorageBuffers();

    void initDescriptorSets() override;

    void updateUniformBuffer(double deltaTime);

    void drawFrame() override;

    void recordComputeCommandBuffer();

    void recordGraphicsCommandBuffer(uint32_t imageIndex) override;
};