#include "vk_loader.h"

#include <fastgltf/core.hpp>
#include <fastgltf/tools.hpp>
#include <fastgltf/glm_element_traits.hpp>

namespace {

std::unexpected<std::string> makeFastGltfError(
        const std::filesystem::path& filePath, 
        fastgltf::Error error) {
    auto errMsg = fastgltf::getErrorMessage(error);
    return std::unexpected(std::format(
        "Failed to load glTF file '{}': {}", 
        filePath.string(), errMsg));
}

}   // namespace

namespace vkutil {

std::expected<std::vector<std::shared_ptr<MeshAsset>>, std::string> loadGltfMeshes(
        const Application& engine,
        const std::filesystem::path& filePath) {
    auto buffer = fastgltf::GltfDataBuffer::FromPath(filePath);
    if (!buffer) {
        return makeFastGltfError(filePath, buffer.error());
    }

    constexpr auto gltfOptions = fastgltf::Options::LoadExternalBuffers;

    fastgltf::Parser parser;
    auto load = parser.loadGltfBinary(
            buffer.get(), filePath.parent_path(), gltfOptions);
    if (!load) {
        return makeFastGltfError(filePath, load.error());
    }

    fastgltf::Asset& asset = load.get();
    std::vector<std::shared_ptr<MeshAsset>> meshes;

    std::vector<uint32_t> indices;
    std::vector<Vertex> vertices;
    for (const auto& mesh : asset.meshes) {
        MeshAsset newmesh;
        newmesh.name = mesh.name;

        // clear the mesh arrays each mesh, we dont want to merge them by error
        indices.clear();
        vertices.clear();

        for (auto&& p : mesh.primitives) {
            GeoSurface newSurface;
            newSurface.startIndex = static_cast<uint32_t>(indices.size());
            newSurface.count = static_cast<uint32_t>(asset.accessors[p.indicesAccessor.value()].count);

            size_t initial_vtx = vertices.size();

            // load indexes
            {
                fastgltf::Accessor& indexaccessor = asset.accessors[p.indicesAccessor.value()];
                indices.reserve(indices.size() + indexaccessor.count);

                fastgltf::iterateAccessor<std::uint32_t>(asset, indexaccessor,
                    [&](std::uint32_t idx) {
                        indices.emplace_back(idx + initial_vtx);
                    });
            }

            // load vertex positions
            {
                auto positionIt = p.findAttribute("POSITION");
                if (positionIt != p.attributes.end()) {
                    fastgltf::Accessor& posAccessor = asset.accessors[positionIt->accessorIndex];
                    vertices.resize(vertices.size() + posAccessor.count);

                    fastgltf::iterateAccessorWithIndex<glm::vec3>(asset, posAccessor,
                        [&](glm::vec3 v, size_t index) {
                            Vertex newvtx;
                            newvtx.position = v;
                            newvtx.normal = { 1, 0, 0 };
                            newvtx.color = glm::vec4 { 1.f };
                            newvtx.uvX = 0;
                            newvtx.uvY = 0;
                            vertices[initial_vtx + index] = newvtx;
                        });
                }
            }

            // load vertex normals
            auto normals = p.findAttribute("NORMAL");
            if (normals != p.attributes.end()) {
                fastgltf::Accessor& normalAccessor = asset.accessors[normals->accessorIndex];
                fastgltf::iterateAccessorWithIndex<glm::vec3>(asset, normalAccessor,
                    [&](glm::vec3 v, size_t index) {
                        vertices[initial_vtx + index].normal = v;
                    });
            }

            // load UVs
            auto uv = p.findAttribute("TEXCOORD_0");
            if (uv != p.attributes.end()) {
                fastgltf::Accessor& uvAccessor = asset.accessors[uv->accessorIndex];
                fastgltf::iterateAccessorWithIndex<glm::vec2>(asset, uvAccessor,
                    [&](glm::vec2 v, size_t index) {
                        vertices[initial_vtx + index].uvX = v.x;
                        vertices[initial_vtx + index].uvY = v.y;
                    });
            }

            // load vertex colors
            auto colors = p.findAttribute("COLOR_0");
            if (colors != p.attributes.end()) {
                fastgltf::Accessor& colorAccessor = asset.accessors[colors->accessorIndex];
                fastgltf::iterateAccessorWithIndex<glm::vec4>(asset, colorAccessor,
                    [&](glm::vec4 v, size_t index) {
                        vertices[initial_vtx + index].color = v;
                    });
            }

            newmesh.surfaces.emplace_back(newSurface);
        }

        // display the vertex normals
        constexpr bool OverrideColors = true;
        if (OverrideColors) {
            for (Vertex& vtx : vertices) {
                vtx.color = glm::vec4(vtx.normal, 1.f);
            }
        }
        newmesh.meshBuffers = engine.uploadMesh(indices, vertices);

        meshes.emplace_back(std::make_shared<MeshAsset>(std::move(newmesh)));
    }

    return meshes;
}

}   // namespace vkutil