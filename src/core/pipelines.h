#pragma once

#include "common/vk_common.h"

#include <string>

namespace vkutil
{

/**
 * @brief Creates a Vulkan shader module from a file.
 */
[[nodiscard]]
vk::raii::ShaderModule loadShaderModule(const std::string& filePath,
                                        const vk::raii::Device& device);

}   // namespace vkutil