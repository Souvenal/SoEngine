#include "frame_data.h"

FrameData::FrameData(const vk::raii::Device& device, const vk::raii::CommandPool& commandPool) {
    vk::CommandBufferAllocateInfo allocInfo {
        .commandPool = commandPool,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = 1
    };
    commandBuffer = std::move(vk::raii::CommandBuffers(device, allocInfo).front());
    presentCompleteSemaphore = vk::raii::Semaphore{device, vk::SemaphoreCreateInfo{}};
    renderFinishedSemaphore = vk::raii::Semaphore{device, vk::SemaphoreCreateInfo{}};
    inFlightFence = vk::raii::Fence{device, vk::FenceCreateInfo{.flags=vk::FenceCreateFlagBits::eSignaled}};
}