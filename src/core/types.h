#pragma once

#include "common/vk_common.h"
#include "common/glm_common.h"

#include <array>

struct Vertex {
    glm::vec3 pos;
    glm::vec3 color;
    glm::vec2 texCoord;

    static vk::VertexInputBindingDescription getBindingDescription();
    static std::array<vk::VertexInputAttributeDescription, 3> getAttributeDescriptions();

    bool operator==(const Vertex& other) const;
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
