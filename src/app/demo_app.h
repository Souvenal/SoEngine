#pragma once

#include "application.h"

struct DemoAppInfo {
};

class DemoApp: public Application {
public:
    DemoApp(
            const std::filesystem::path& appDir,
            const Window* window,
            const AppInfo& appInfo,
            const DemoAppInfo& info);
    ~DemoApp();

    void onInit() override;
    // void onUpdate(double deltaTime) override;
    // void onInputEvent(const InputEvent& event) override;
    void onShutdown() override;

    // void recreateSwapchain() override;

protected:
    DemoAppInfo demoAppInfo;

    vk::raii::CommandBuffers        graphicsCommandBuffers { nullptr };
    vk::raii::CommandBuffers        computeCommandBuffers  { nullptr };

    vk::raii::DescriptorSetLayout   drawImageDescriptorSetLayout { nullptr };
    AllocatedImage                  drawImage;
    std::vector<vk::raii::DescriptorSet> drawImageDescriptorSets;

    vk::raii::PipelineLayout        gradientPipelineLayout { nullptr };
    vk::raii::Pipeline              gradientPipeline { nullptr };

protected:
    void initCommandBuffers() override;

    /**
     * @brief Initializes the swap chain and the draw image
     */
    void initSwapchain() override;

    void initDescriptorAllocator() override;

    void initDescriptorSetLayouts() override;

    void initPipelines() override;

    void initBackgroundPipelines();

    void initDescriptorSets() override;

    void drawFrame() override;

    void drawBackground(const vk::raii::CommandBuffer& commandBuffer) override;

    // void drawScene(const vk::raii::CommandBuffer& commandBuffer, uint32_t imageIndex) override;
};
