#pragma once

#include "common/vk_common.h"
#include "common/glm_common.h"

#include <array>
#include <optional>

struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily; // support drawing commands
    std::optional<uint32_t> presentFamily;  // support presentation
    std::optional<uint32_t> computeFamily;  // support compute operations
    std::optional<uint32_t> transferFamily; // support transfer operations

    [[nodiscard]] bool isComplete() const;
};

class MemoryAllocator {
public:
    VmaAllocator allocator  { nullptr };

public:
    MemoryAllocator() = default;
    MemoryAllocator(const vk::Instance& instance,
                    const vk::PhysicalDevice& physicalDevice,
                    const vk::Device& device);
    MemoryAllocator(const MemoryAllocator&) = delete;
    MemoryAllocator& operator=(const MemoryAllocator&) = delete;
    MemoryAllocator(MemoryAllocator&& rhs) noexcept;
    MemoryAllocator& operator=(MemoryAllocator&& rhs) noexcept;

    ~MemoryAllocator();
};

enum class MemoryType {
    DeviceLocal,
    HostVisible
};

class AllocatedBuffer {
public:
    vk::Buffer          buffer          { nullptr };
    VmaAllocation       allocation      { nullptr };
    VmaAllocationInfo   allocationInfo  {};

    AllocatedBuffer() = default;
    AllocatedBuffer(const AllocatedBuffer&) = delete;
    AllocatedBuffer& operator=(const AllocatedBuffer&) = delete;
    AllocatedBuffer(AllocatedBuffer&& rhs);
    AllocatedBuffer& operator=(AllocatedBuffer&& rhs);
    AllocatedBuffer(VmaAllocator allocator,
                    const QueueFamilyIndices& indices,
                    vk::DeviceSize size,
                    vk::BufferUsageFlags usage,
                    MemoryType memoryType); 
    ~AllocatedBuffer();

    template<typename T>
    void write(const T* src, size_t count, vk::DeviceSize offset = 0) {
        void* data = map(offset, sizeof(T) * count);
        memcpy(data, src, sizeof(T) * count);
        unmap();
    }

    void* map(vk::DeviceSize offset = 0, vk::DeviceSize size = vk::WholeSize);
    void unmap();

private:
    VmaAllocator allocator      { nullptr };
};

class AllocatedImage {
public:
    vk::Image       image       { nullptr };
    vk::ImageView   imageView   { nullptr };
    VmaAllocation   allocation  { nullptr };
    vk::Extent3D    imageExtent {};
    vk::Format      imageFormat {};

    AllocatedImage() = default;
    AllocatedImage(const AllocatedImage&) = delete;
    AllocatedImage& operator=(const AllocatedImage&) = delete;
    AllocatedImage(AllocatedImage&& rhs);
    AllocatedImage& operator=(AllocatedImage&& rhs);
    AllocatedImage(const vk::Device& device,
                  VmaAllocator allocator,
                  vk::Extent3D extent,
                  uint32_t mipLevels,
                  vk::SampleCountFlagBits numSamples,
                  vk::Format format,
                  vk::ImageTiling tiling,
                  vk::ImageUsageFlags usage,
                  vk::ImageAspectFlags aspectFlags,
                  MemoryType memoryType);
    ~AllocatedImage();

private:
    vk::Device      device      { nullptr };
    VmaAllocator    allocator   { nullptr };
};

namespace vkutil
{

[[nodiscard]]
QueueFamilyIndices findQueueFamilies(const vk::raii::PhysicalDevice& physicalDevice,
                                     const vk::raii::SurfaceKHR& surface);

void copyAllocatedBuffer(const vk::raii::CommandBuffer& commandBuffer,
                         const AllocatedBuffer& src,
                         const AllocatedBuffer& dst,
                         vk::DeviceSize size);

}   // namespace vkutil