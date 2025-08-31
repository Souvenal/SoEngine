#include "types.h"

vk::VertexInputBindingDescription Vertex::getBindingDescription() {
    vk::VertexInputBindingDescription bindingDescription {
        .binding = 0, // index of the binding in the array of bindings
        .stride = sizeof(Vertex), // the number of bytes from one entry to the next
        .inputRate = vk::VertexInputRate::eVertex
    };

    return bindingDescription;
}

std::array<vk::VertexInputAttributeDescription, 3> Vertex::getAttributeDescriptions() {
    return {
        vk::VertexInputAttributeDescription( 0, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, pos) ),
        vk::VertexInputAttributeDescription( 1, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, color) ),
        vk::VertexInputAttributeDescription( 2, 0, vk::Format::eR32G32Sfloat, offsetof(Vertex, texCoord) )
    };
}

bool Vertex::operator==(const Vertex& other) const {
    return
        pos == other.pos &&
        color == other.color &&
        texCoord == other.texCoord;
}