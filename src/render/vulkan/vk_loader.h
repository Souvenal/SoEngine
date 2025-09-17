#pragma once

#include "stb_image.h"

#include "vk_types.h"
#include "common/error.h"
#include "app/application.h"

struct GeoSurface {
    uint32_t startIndex;
    uint32_t count;
};

struct MeshAsset {
    std::string name;

    std::vector<GeoSurface> surfaces;
    GPUMeshBuffers meshBuffers;
};

namespace vkutil {

[[nodiscard]]
std::expected<std::vector<std::shared_ptr<MeshAsset>>, std::string> loadGltfMeshes(
        const Application& engine,
        const std::filesystem::path& filePath);

}   // namespace vkutil