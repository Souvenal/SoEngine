#pragma once

#include "common/glm_common.h"
#include "common/vk_common.h"



class DescriptorLayoutBuilder {
public:
	DescriptorLayoutBuilder() = default;

	void addBinding(uint32_t binding, vk::DescriptorType type, vk::ShaderStageFlags shaderStages);
	void clear();

	[[nodiscard]]
	vk::raii::DescriptorSetLayout build(
		const vk::raii::Device& device,
		void* pNext = nullptr,
		vk::DescriptorSetLayoutCreateFlags flags = {}) const;

private:
	std::vector<vk::DescriptorSetLayoutBinding> bindings {};
};

class DescriptorAllocator {
public:
	struct PoolSizeRatio {
		vk::DescriptorType type;
		float ratio;
		
		PoolSizeRatio() = delete;
		PoolSizeRatio(vk::DescriptorType type, float ratio)
			: type(type), ratio(ratio) {}
	};

	DescriptorAllocator() = default;
	DescriptorAllocator(const vk::raii::Device& device,
						uint32_t maxSets,
						std::span<PoolSizeRatio> poolSizeRatios);
	
	void clearDescriptors();
	std::vector<vk::raii::DescriptorSet> allocate(const vk::raii::Device& device,
									 const vk::raii::DescriptorSetLayout& layout,
									 const uint32_t layoutCount = 1);

	vk::DescriptorPool getPool() const;

private:
	vk::raii::DescriptorPool pool	{ nullptr };
};