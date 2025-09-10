#pragma once

#include "application.h"
#include "common/vk_common.h"
#include "resource/model_object.h"

#include <ktxvulkan.h>

struct ModelAppInfo {
    vk::SampleCountFlagBits msaaSamples { vk::SampleCountFlagBits::e1 };
    const uint32_t maxObjects { 3 };
    const std::string modelPath { "models/viking_room.obj" };
    const std::string texturePath { "models/viking_room/textures/viking_room.ktx2" };
};

class ModelApp : public Application {
public:
    ModelApp(
            const std::filesystem::path& appDir,
            const AppInfo& appInfo,
            const ModelAppInfo& info);
    ~ModelApp();

    void onInit(const Window* window) override;
    void onUpdate(double deltaTime) override;
    void onRender() override;
    // void onInputEvent(const InputEvent& event) override;
    void onShutdown() override;

    void recreateSwapchain() override;

public:
    struct Vertex {
        glm::vec3 pos;
        glm::vec3 color;
        glm::vec2 texCoord;

        static vk::VertexInputBindingDescription getBindingDescription();
        static std::array<vk::VertexInputAttributeDescription, 3> getAttributeDescriptions();

        bool operator==(const Vertex& other) const;
    };

    struct UniformBufferObject {
        alignas(16) glm::mat4 model;
        alignas(16) glm::mat4 view;
        alignas(16) glm::mat4 proj;
    };

protected:
    ModelAppInfo modelAppInfo;

    vk::raii::DescriptorSetLayout       drawImageDescriptorSetLayout { nullptr };

    AllocatedImage  colorImage {};
    AllocatedImage  depthImage {};

    uint32_t                mipLevels { 1 };
    ktxVulkanTexture        texture;
    vk::raii::ImageView     textureImageView { nullptr };
    vk::raii::Sampler       textureSampler { nullptr };

    std::vector<Vertex>     vertices {};
    std::vector<uint32_t>   indices {};
    AllocatedBuffer         vertexBuffer {};
    AllocatedBuffer         indexBuffer {};

    std::vector<vk::raii::CommandBuffer>    graphicsCommandBuffers;

    std::vector<ModelObject> modelObjects;

    float minX { std::numeric_limits<float>::max() };
    float maxX { std::numeric_limits<float>::min() };
    float minY { std::numeric_limits<float>::max() };
    float maxY { std::numeric_limits<float>::min() };
    float minZ { std::numeric_limits<float>::max() };
    float maxZ { std::numeric_limits<float>::min() };

protected:
    void initCommandBuffers() override;
    
    void initDescriptorAllocator() override;

    void initDescriptorSetLayouts() override;

    void initRenderPass() override;

    void initGraphicsPipeline() override;

    void initColorResources();

    void initDepthResources();

    void initFramebuffers() override;

    void loadModel(const std::string& modelName);

    void setupModelObjects();

    void initVertexBuffer();

    void initIndexBuffer();

    void initTextureImage();

    void initTextureImageView();

    void initTextureSampler();

    void initUniformBuffers();

    void initDescriptorSets() override;

    void buildCapabilitiesSummary() override;

    void logCapabilitiesSummary() const override;

    void updateUniformBuffer(double deltaTime);

    void drawFrame() override;

    void recordGraphicsCommandBuffer(uint32_t imageIndex) override;
};

template<>
struct std::hash<ModelApp::Vertex> {
    size_t operator()(ModelApp::Vertex const& vertex) const noexcept {
        return ((hash<glm::vec3>{}(vertex.pos)) ^
                ((hash<glm::vec3>{}(vertex.color) << 1) >> 1) ^
                (hash<glm::vec2>{}(vertex.texCoord) << 1)
        );
    }
};