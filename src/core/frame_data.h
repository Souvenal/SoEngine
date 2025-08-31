#pragma once

#include "common/vk_common.h"
#include "utils/deletion_queue.h"

struct FrameData {
    FrameData(const vk::raii::Device& device, const vk::raii::CommandPool& commandPool);

    vk::raii::Semaphore presentCompleteSemaphore { nullptr };
    vk::raii::Semaphore renderFinishedSemaphore { nullptr };
    vk::raii::Fence inFlightFence { nullptr };

    vk::raii::CommandBuffer commandBuffer { nullptr };

    DeletionQueue deletionQueue;
};