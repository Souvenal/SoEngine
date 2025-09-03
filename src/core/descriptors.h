#pragma once

#include "common/glm_common.h"
#include "common/vk_common.h"

struct UniformBufferObject {
	alignas(16) glm::mat4 model;
	alignas(16) glm::mat4 view;
	alignas(16) glm::mat4 proj;
};

struct DescriptorLayoutBuilder {
	std::vector<vk::DescriptorSetLayoutBinding> bindings {};

	void addBinding(uint32_t binding, vk::DescriptorType type, vk::ShaderStageFlags shaderStages);
	void clear();

	[[nodiscard]]
	vk::raii::DescriptorSetLayout build(
		const vk::raii::Device& device,
		void* pNext = nullptr,
		vk::DescriptorSetLayoutCreateFlags flags = {}) const;
};

struct DescriptorAllocator {
	struct PoolSizeRatio {
		vk::DescriptorType type;
		float ratio;
		
		PoolSizeRatio() = delete;
		PoolSizeRatio(vk::DescriptorType type, float ratio)
			: type(type), ratio(ratio) {}
	};

	vk::raii::DescriptorPool pool	{ nullptr };

	DescriptorAllocator() = default;
	DescriptorAllocator(const vk::raii::Device& device,
						uint32_t maxSets,
						std::span<PoolSizeRatio> poolSizeRatios);
	
	void clearDescriptors();
	std::vector<vk::raii::DescriptorSet> allocate(const vk::raii::Device& device,
									 const vk::raii::DescriptorSetLayout& layout,
									 const uint32_t layoutCount = 1);
};