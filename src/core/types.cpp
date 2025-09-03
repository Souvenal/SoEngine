#include "types.h"

#include "initializers.h"

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
    allocator(rhs.allocator)
{
    rhs.allocator = nullptr;
}

MemoryAllocator& MemoryAllocator::operator=(MemoryAllocator&& rhs) noexcept {
    allocator = rhs.allocator;
    rhs.allocator = nullptr;
    return *this;
}

MemoryAllocator::~MemoryAllocator() {
    if (allocator != nullptr) {
        vmaDestroyAllocator(allocator);
    }
    allocator = nullptr;
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

AllocatedBuffer::AllocatedBuffer(AllocatedBuffer&& rhs):
    buffer(rhs.buffer), allocation(rhs.allocation), allocationInfo(rhs.allocationInfo), allocator(rhs.allocator)
{
    rhs.buffer = nullptr;
    rhs.allocation = nullptr;
    rhs.allocationInfo = {};
    rhs.allocator = nullptr;
}
AllocatedBuffer& AllocatedBuffer::operator=(AllocatedBuffer&& rhs) {
    buffer = rhs.buffer;
    allocation = rhs.allocation;
    allocationInfo = rhs.allocationInfo;
    allocator = rhs.allocator;
    rhs.buffer = nullptr;
    rhs.allocation = nullptr;
    rhs.allocationInfo = {};
    rhs.allocator = nullptr;
    return *this;
}

AllocatedBuffer::~AllocatedBuffer() {
    if (allocator && buffer && allocation != VK_NULL_HANDLE)
        vmaDestroyBuffer(allocator, buffer, allocation);
    buffer = nullptr;
    allocation = nullptr;
    allocator = nullptr;
}

void* AllocatedBuffer::map(vk::DeviceSize offset, vk::DeviceSize size) {
    void* data = nullptr;
    vmaMapMemory(allocator, allocation, &data);
    return static_cast<char*>(data) + offset;
}

void AllocatedBuffer::unmap() {
    vmaUnmapMemory(allocator, allocation);
}

AllocatedImage::AllocatedImage(const vk::Device& device,
                               VmaAllocator allocator,
                               vk::Extent3D extent,
                               uint32_t mipLevels,
                               vk::SampleCountFlagBits numSamples,
                               vk::Format format,
                               vk::ImageTiling tiling,
                               vk::ImageUsageFlags usage,
                               MemoryType memoryType):
    device(device), allocator(allocator)
{
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

    auto imageViewInfo = vkinit::imageViewCreateInfo(image, format, vk::ImageAspectFlagBits::eColor, mipLevels);
    imageView = device.createImageView(imageViewInfo);
}

AllocatedImage::AllocatedImage(AllocatedImage&& rhs):
    device(rhs.device), allocator(rhs.allocator),
    image(rhs.image), imageView(rhs.imageView),
    allocation(rhs.allocation), imageExtent(rhs.imageExtent),
    imageFormat(rhs.imageFormat)
{
    rhs.device = nullptr;
    rhs.allocator = nullptr;
    rhs.image = nullptr;
    rhs.imageView = nullptr;
    rhs.allocation = nullptr;
    rhs.imageExtent = {};
    rhs.imageFormat = {};
}

AllocatedImage& AllocatedImage::operator=(AllocatedImage&& rhs) {
    device = rhs.device;
    allocator = rhs.allocator;
    image = rhs.image;
    imageView = rhs.imageView;
    allocation = rhs.allocation;
    imageExtent = rhs.imageExtent;
    imageFormat = rhs.imageFormat;
    rhs.device = nullptr;
    rhs.allocator = nullptr;
    rhs.image = nullptr;
    rhs.imageView = nullptr;
    rhs.allocation = nullptr;
    rhs.imageExtent = {};
    rhs.imageFormat = {};
    return *this;
}

AllocatedImage::~AllocatedImage() {
    if (device && imageView)
        device.destroyImageView(imageView);
    if (allocator != nullptr && image && allocation)
        vmaDestroyImage(allocator, image, allocation);
    device = nullptr;
    allocator = nullptr;
    image = nullptr;
    imageView = nullptr;
    allocation = nullptr;
    imageExtent = {};
    imageFormat = {};
}

vk::VertexInputBindingDescription Vertex::getBindingDescription() {
    vk::VertexInputBindingDescription bindingDescription {
        .binding = 0, // index of the binding in the array of bindings
        .stride = sizeof(Vertex), // the number of bytes from one entry to the next
        .inputRate = vk::VertexInputRate::eVertex
    };

    return bindingDescription;
}

std::array<vk::VertexInputAttributeDescription, 3> Vertex::getAttributeDescriptions() {
    return {
        vk::VertexInputAttributeDescription( 0, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, pos) ),
        vk::VertexInputAttributeDescription( 1, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, color) ),
        vk::VertexInputAttributeDescription( 2, 0, vk::Format::eR32G32Sfloat, offsetof(Vertex, texCoord) )
    };
}

bool Vertex::operator==(const Vertex& other) const {
    return
        pos == other.pos &&
        color == other.color &&
        texCoord == other.texCoord;
}

namespace vkutil
{

void copyAllocatedBuffer(const vk::raii::CommandBuffer& commandBuffer,
                         const AllocatedBuffer& src,
                         const AllocatedBuffer& dst,
                         vk::DeviceSize size)
{
    vk::BufferCopy copyRegion{
        .srcOffset = 0,
        .dstOffset = 0,
        .size = size
    };
    commandBuffer.copyBuffer(src.buffer, dst.buffer, copyRegion);
}

}   // namespace vkutil