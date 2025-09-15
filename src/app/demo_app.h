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
    struct ComputePushConstants {
        glm::vec4 data1;
        glm::vec4 data2;
        glm::vec4 data3;
        glm::vec4 data4;
    };

    struct ComputeEffect {
        std::string name;

        vk::PipelineLayout layout;
        vk::raii::Pipeline pipeline { nullptr };

        ComputePushConstants data;

        ComputeEffect() = default;
        ComputeEffect(ComputeEffect&& other) noexcept
            : name(std::move(other.name))
            , layout(std::exchange(other.layout, nullptr))
            , pipeline(std::move(other.pipeline))
            , data(std::move(other.data)) { }

        ComputeEffect& operator=(ComputeEffect&& other) noexcept {
            if (this != &other) {
                name = std::move(other.name);
                layout = std::exchange(other.layout, nullptr);
                pipeline = std::move(other.pipeline);
                data = std::move(other.data);
            }
            return *this;
        }

        ComputeEffect(const ComputeEffect&) = delete;
        ComputeEffect& operator=(const ComputeEffect&) = delete;
    };


    DemoAppInfo demoAppInfo;

    vk::raii::CommandBuffers        graphicsCommandBuffers { nullptr };
    vk::raii::CommandBuffers        computeCommandBuffers  { nullptr };

    vk::raii::PipelineLayout        gradientPipelineLayout { nullptr };
    // vk::raii::Pipeline              gradientPipeline { nullptr };

    std::vector<ComputeEffect>       backgroundEffects {};
    int                              currentBackgroundEffect {0};

protected:
    void initCommandBuffers() override;

    void initDescriptorAllocator() override;

    void initDescriptorSetLayouts() override;

    void initPipelines() override;

    void initBackgroundPipelines();

    void initDescriptorSets() override;

    void updateImGui() override;

    void drawFrame() override;

    void drawBackground(const vk::raii::CommandBuffer& commandBuffer) override;

    // void drawScene(const vk::raii::CommandBuffer& commandBuffer, uint32_t imageIndex) override;
};
