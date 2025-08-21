#pragma once

import vulkan_hpp;

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>

#include <array>

struct Vertex {
    glm::vec3 pos;
    glm::vec3 color;
    glm::vec2 texCoord;

    static vk::VertexInputBindingDescription getBindingDescription() {
        vk::VertexInputBindingDescription bindingDescription {
            .binding = 0, // index of the binding in the array of bindings
            .stride = sizeof(Vertex), // the number of bytes from one entry to the next
            .inputRate = vk::VertexInputRate::eVertex
        };

        return bindingDescription;
    }

    static std::array<vk::VertexInputAttributeDescription, 3> getAttributeDescriptions() {
        return {
            vk::VertexInputAttributeDescription( 0, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, pos) ),
            vk::VertexInputAttributeDescription( 1, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, color) ),
            vk::VertexInputAttributeDescription( 2, 0, vk::Format::eR32G32Sfloat, offsetof(Vertex, texCoord) )
        };
    }

    bool operator==(const Vertex& other) const {
        return
            pos == other.pos &&
            color == other.color &&
            texCoord == other.texCoord;
    }
};

template<>
struct std::hash<Vertex> {
    size_t operator()(Vertex const& vertex) const noexcept {
        return ((hash<glm::vec3>{}(vertex.pos)) ^
                ((hash<glm::vec3>{}(vertex.color) << 1) >> 1) ^
                (hash<glm::vec2>{}(vertex.texCoord) << 1)
        );
    }
};
