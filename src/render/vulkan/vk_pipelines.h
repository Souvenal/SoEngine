#pragma once

#include "common/vk_common.h"

#include <string>

namespace vkutil
{

/**
 * @brief Creates a Vulkan shader module from a file.
 */
[[nodiscard]]
vk::raii::ShaderModule loadShaderModule(const vk::raii::Device& device,
                                        const std::string& filePath);

}   // namespace vkutil