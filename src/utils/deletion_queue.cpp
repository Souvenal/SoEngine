#include "deletion_queue.h"

void DeletionQueue::pushFunction(std::function<void()>&& function) {
    deletors.emplace_back(std::move(function));
}

void DeletionQueue::flush() {
    for (auto it = deletors.rbegin(); it != deletors.rend(); ++it) {
        (*it)();
    }
    deletors.clear();
}
