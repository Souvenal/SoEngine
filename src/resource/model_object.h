#pragma once

#include "common/glm_common.h"
#include "core/types.h"

#include <vector>
#include <limits>

struct ModelObject {
    // Transform properties
    glm::vec3 position  { 0.0f, 0.0f, 0.0f };
    glm::vec3 rotation  { 0.0f, 0.0f, 0.0f };
    glm::vec3 scale     { 1.0f, 1.0f, 1.0f };
    float minX { std::numeric_limits<float>::max() };
    float maxX { std::numeric_limits<float>::min() };
    float minY { minX };
    float maxY { maxX };
    float minZ { minX };
    float maxZ { maxX };

    // Uniform buffer for this object (one per frame in flight)
    // std::vector<vk::raii::Buffer>       uniformBuffers;
    // std::vector<vk::raii::DeviceMemory> uniformBuffersMemory;
    // std::vector<void*>                  uniformBuffersMapped;
    std::vector<AllocatedBuffer> uniformBuffers;

    // Descriptor sets for this object (one per frame in flight)
    std::vector<vk::raii::DescriptorSet> descriptorSets;

    // Calculate model matrix based on position, rotation, and scale
    glm::mat4 getModelMatrix() const {
        glm::mat4 model = glm::mat4(1.0f);
        model = glm::translate(model, position);
        model = glm::rotate(model, rotation.x, glm::vec3(1.0f, 0.0f, 0.0f));
        model = glm::rotate(model, rotation.y, glm::vec3(0.0f, 1.0f, 0.0f));
        model = glm::rotate(model, rotation.z, glm::vec3(0.0f, 0.0f, 1.0f));
        model = glm::scale(model, scale);
        return model;
    }
};