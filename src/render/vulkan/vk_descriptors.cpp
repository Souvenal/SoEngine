#include "vk_descriptors.h"

void DescriptorLayoutBuilder::addBinding(
    uint32_t binding,
    vk::DescriptorType type,
    vk::ShaderStageFlags shaderStages
) {
    vk::DescriptorSetLayoutBinding layoutBinding = {
        .binding = binding,
        .descriptorType = type,
        .descriptorCount = 1,
        .stageFlags = shaderStages
    };
	bindings.emplace_back(std::move(layoutBinding));
}

void DescriptorLayoutBuilder::clear() {
	bindings.clear();
}

vk::raii::DescriptorSetLayout DescriptorLayoutBuilder::build(
        const vk::raii::Device& device,
        void* pNext,
        vk::DescriptorSetLayoutCreateFlags flags) const {
    vk::DescriptorSetLayoutCreateInfo createInfo = {
        .pNext = pNext,
        .flags = flags,
        .bindingCount = static_cast<uint32_t>(bindings.size()),
        .pBindings = bindings.data(),
    };

    return device.createDescriptorSetLayout(createInfo);
}

DescriptorAllocator::DescriptorAllocator(
        const vk::raii::Device& device,
        uint32_t maxSets,
        std::span<PoolSizeRatio> poolSizeRatios) {
	// Create the descriptor pool
	std::vector<vk::DescriptorPoolSize> sizes;
	sizes.reserve(poolSizeRatios.size());
	for (const auto& ratio : poolSizeRatios) {
		sizes.emplace_back(
			ratio.type,
			static_cast<uint32_t>(maxSets * ratio.ratio)
		);
	}

	vk::DescriptorPoolCreateInfo poolInfo = {
        .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
		.maxSets = maxSets,
		.poolSizeCount = static_cast<uint32_t>(sizes.size()),
		.pPoolSizes = sizes.data()
	};

	pool = device.createDescriptorPool(poolInfo);
}

void DescriptorAllocator::clearDescriptors() {
	pool.reset();
}

vk::raii::DescriptorSets DescriptorAllocator::allocate(
    const vk::raii::Device& device,
    const vk::raii::DescriptorSetLayout& layout,
    const uint32_t layoutCount
) {
    std::vector<vk::DescriptorSetLayout> layouts(layoutCount, *layout);
    vk::DescriptorSetAllocateInfo allocInfo = {
        .descriptorPool = *pool,
        .descriptorSetCount = static_cast<uint32_t>(layoutCount),
        .pSetLayouts = layouts.data()
    };

    return vk::raii::DescriptorSets(device, allocInfo);
}

vk::DescriptorPool DescriptorAllocator::getPool() const {
    return *pool;
}