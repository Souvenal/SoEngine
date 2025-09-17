#include "vk_types.h"

#include "vk_initializers.h"
#include "utils/logging.h"

#define VMA_IMPLEMENTATION
#include <vma/vk_mem_alloc.h>

#include <fstream>

bool QueueFamilyIndices::isComplete() const {
    return
        graphicsFamily.has_value() &&
        presentFamily.has_value() &&
        computeFamily.has_value() &&
        transferFamily.has_value();
}

MemoryAllocator::MemoryAllocator(const vk::Instance& instance,
                const vk::PhysicalDevice& physicalDevice,
                const vk::Device& device)
{
    VmaAllocatorCreateInfo createInfo {
        .flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
        .physicalDevice = physicalDevice,
        .device = device,
        .instance = instance,
    };

    if (vmaCreateAllocator(&createInfo, &allocator) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create VMA allocator!");
    }
}

MemoryAllocator::MemoryAllocator(MemoryAllocator&& rhs) noexcept:
    allocator(std::exchange(rhs.allocator, nullptr)) { }

MemoryAllocator& MemoryAllocator::operator=(MemoryAllocator&& rhs) noexcept {
    if (this != &rhs) {
        std::swap(allocator, rhs.allocator);
    }
    return *this;
}

MemoryAllocator::~MemoryAllocator() {
    if (allocator != nullptr) {
        vmaDestroyAllocator(allocator);
    }
}

AllocatedBuffer::AllocatedBuffer(VmaAllocator allocator,
                                const QueueFamilyIndices& indices,
                                vk::DeviceSize size,
                                vk::BufferUsageFlags usage,
                                MemoryType memoryType):
    allocator(allocator)
{
    auto bufferInfo = VkBufferCreateInfo(vkinit::bufferCreateInfo(size, usage, indices));
    auto vmaAllocCreateInfo = vkinit::vmaAllocationCreateInfo(memoryType);

    VkBuffer _buffer;
    if (vmaCreateBuffer(allocator, &bufferInfo, &vmaAllocCreateInfo, &_buffer, &allocation, &allocationInfo) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create buffer!");
    }
    buffer = vk::Buffer(_buffer);
}

AllocatedBuffer::AllocatedBuffer(AllocatedBuffer&& rhs) noexcept
    : buffer(std::exchange(rhs.buffer, nullptr)),
      allocation(std::exchange(rhs.allocation, nullptr)),
      allocationInfo(std::exchange(rhs.allocationInfo, {})),
      allocator(std::exchange(rhs.allocator, nullptr))
{}
AllocatedBuffer& AllocatedBuffer::operator=(AllocatedBuffer&& rhs) {
    if (this != &rhs) {
        std::swap(buffer, rhs.buffer);
        std::swap(allocation, rhs.allocation);
        std::swap(allocationInfo, rhs.allocationInfo);
        std::swap(allocator, rhs.allocator);
    }
    return *this;
}

AllocatedBuffer::~AllocatedBuffer() {
    if (allocator && buffer && allocation != VK_NULL_HANDLE)
        vmaDestroyBuffer(allocator, buffer, allocation);
}

void* AllocatedBuffer::map(vk::DeviceSize offset, vk::DeviceSize size) {
    void* data = nullptr;
    vmaMapMemory(allocator, allocation, &data);
    return static_cast<char*>(data) + offset;
}

void AllocatedBuffer::unmap() {
    vmaUnmapMemory(allocator, allocation);
}

AllocatedImage::AllocatedImage(
        const vk::Device& device,
        VmaAllocator allocator,
        vk::Extent3D extent,
        uint32_t mipLevels,
        vk::SampleCountFlagBits numSamples,
        vk::Format format,
        vk::ImageTiling tiling,
        vk::ImageUsageFlags usage,
        vk::ImageAspectFlags aspectFlags,
        MemoryType memoryType)
    : device(device), allocator(allocator) {
    imageExtent = extent;
    imageFormat = format;

    auto imageInfo = VkImageCreateInfo(vkinit::imageCreateInfo(format, extent, mipLevels, numSamples, tiling, usage));
    auto vmaAllocCreateInfo = vkinit::vmaAllocationCreateInfo(memoryType);

    VkImage _image;
    if (vmaCreateImage(allocator, &imageInfo, &vmaAllocCreateInfo,
        &_image, &allocation, nullptr) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create image");
    }
    image = _image;

    auto imageViewInfo = vkinit::imageViewCreateInfo(image, format, aspectFlags, mipLevels);
    imageView = device.createImageView(imageViewInfo);
}

AllocatedImage::AllocatedImage(AllocatedImage&& rhs):
    device(std::exchange(rhs.device, nullptr)),
    allocator(std::exchange(rhs.allocator, nullptr)),
    image(std::exchange(rhs.image, nullptr)),
    imageView(std::exchange(rhs.imageView, nullptr)),
    allocation(std::exchange(rhs.allocation, nullptr)),
    imageExtent(std::exchange(rhs.imageExtent, {})),
    imageFormat(std::exchange(rhs.imageFormat, {})) { }

AllocatedImage& AllocatedImage::operator=(AllocatedImage&& rhs) {
    if (this != &rhs) {
        std::swap(device, rhs.device);
        std::swap(allocator, rhs.allocator);
        std::swap(image, rhs.image);
        std::swap(imageView, rhs.imageView);
        std::swap(allocation, rhs.allocation);
        std::swap(imageExtent, rhs.imageExtent);
        std::swap(imageFormat, rhs.imageFormat);
    }
    return *this;
}

AllocatedImage::~AllocatedImage() {
    if (device && imageView)
        device.destroyImageView(imageView);
    if (allocator != nullptr && image && allocation)
        vmaDestroyImage(allocator, image, allocation);
}

namespace vkutil
{

QueueFamilyIndices findQueueFamilies(
        const vk::raii::PhysicalDevice& physicalDevice,
        const vk::raii::SurfaceKHR& surface) {
    QueueFamilyIndices indices;
    const auto queueFamilyProperties = physicalDevice.getQueueFamilyProperties();

    // First pass: try to grab specialized queues (compute-only / transfer-only) while also
    // recording graphics & present. We prefer dedicated queues to reduce contention.
    for (uint32_t i = 0; i < queueFamilyProperties.size(); ++i) {
        const auto& qfp = queueFamilyProperties[i];

        bool graphicsSupport = (qfp.queueFlags & vk::QueueFlagBits::eGraphics) != vk::QueueFlagBits{};
        bool computeSupport  = (qfp.queueFlags & vk::QueueFlagBits::eCompute)  != vk::QueueFlagBits{};
        bool transferSupport = (qfp.queueFlags & vk::QueueFlagBits::eTransfer) != vk::QueueFlagBits{};
        VkBool32 presentSupport = physicalDevice.getSurfaceSupportKHR(i, surface);

        // Record graphics family
        if (!indices.graphicsFamily && graphicsSupport)
            indices.graphicsFamily = i;

        // Record present family
        if (!indices.presentFamily && presentSupport)
            indices.presentFamily = i;

        // Prefer a compute-only queue (compute without graphics)
        if (!indices.computeFamily && computeSupport && !graphicsSupport)
            indices.computeFamily = i;

        // Prefer a transfer-only queue (transfer without graphics)
        if (!indices.transferFamily && transferSupport && !graphicsSupport && !computeSupport)
            indices.transferFamily = i;

        // Early out if everything already found
        if (indices.isComplete())
            break;
    }

    // If no dedicated compute, fallback to graphics (do NOT force different index just因为不同)
    if (!indices.computeFamily && indices.graphicsFamily)
        indices.computeFamily = indices.graphicsFamily;

    // If no dedicated transfer, fallback likewise
    if (!indices.transferFamily && indices.graphicsFamily)
        indices.transferFamily = indices.graphicsFamily;

    if (!indices.isComplete()) {
        throw std::runtime_error("Failed to find required queue families (graphics/present/compute/transfer).");
    }

    return indices;
}

void copyAllocatedBuffer(
        const vk::raii::CommandBuffer& commandBuffer,
        const AllocatedBuffer& src,
        const AllocatedBuffer& dst,
        vk::BufferCopy copyRegion) {
    commandBuffer.copyBuffer(src.buffer, dst.buffer, copyRegion);
}

}   // namespace vkutil