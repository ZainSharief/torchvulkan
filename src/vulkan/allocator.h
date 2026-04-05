#pragma once
#include <torch/extension.h>
#include <c10/core/Allocator.h>
#include <vector>
#include <mutex>

#include "vulkan_context.h"
#include "memory.h"

struct VulkanAllocator : public c10::Allocator {
    mutable std::vector<VulkanBuffer*> deleteQueue;
    mutable std::mutex mutex_; // thread safety

    static const uint32_t SYNC_THRESHOLD = 200;
    static const float VRAM_USAGE_LIMIT = 0.95; // leave headroom for driver allocations
    mutable std::atomic<size_t> allocated_bytes{0};
    mutable std::atomic<uint32_t> num_allocations_since_sync{SYNC_THRESHOLD};
    mutable std::atomic<size_t> vram_limit{0};

    c10::DataPtr allocate(size_t nbytes) override;
    c10::DataPtr allocate(size_t nbytes) const {
        return const_cast<VulkanAllocator*>(this)->allocate(nbytes);
    }

    void clearResources() const;
    static void deleter(void* ptr);

    void copy_data(void* dest, const void* src, std::size_t count) const override; // defaults to host-to-device
    void copy_host_to_device(void* dest, const void* src, std::size_t count) const;
    void copy_device_to_host(void* dest, const void* src, std::size_t count) const;
    void copy_device_to_device(void* dest, const void* src, std::size_t count) const;

    VulkanBuffer* out_of_memory_buffer(size_t size, MemoryUsage usage) const;
    void sync_memory_budget(size_t size) const;
};

extern VulkanAllocator globalVulkanAllocator;