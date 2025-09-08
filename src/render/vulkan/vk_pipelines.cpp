#include "vk_pipelines.h"

#include "utils/logging.h"

#include <fstream>

namespace {
std::vector<char> readFile(const std::string& filename) {
    std::ifstream file (filename, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file at " + filename);
    }

    const size_t fileSize = static_cast<size_t>(file.tellg());
    std::vector<char> buffer(fileSize);

    file.seekg(0, std::ios::beg);
    file.read(buffer.data(), static_cast<std::streamsize>(fileSize));
    file.close();

    return buffer;
}
}   // namespace

namespace vkutil
{

vk::raii::ShaderModule loadShaderModule(const vk::raii::Device& device,
                                        const std::string& filePath)
{
    auto code = readFile(filePath);

    vk::ShaderModuleCreateInfo createInfo {
        .codeSize = code.size() * sizeof(char),
        .pCode = reinterpret_cast<const uint32_t*>(code.data())
    };

    return device.createShaderModule(createInfo);
}

} // namespace vkutil