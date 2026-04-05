#pragma once
#include <vector>
#include <mutex>
#include <stdexcept>
#include <map>
#include <array>

#include "volk.h"
#include "memory.h"

#define VK_CHECK(x) \
    if (x != VK_SUCCESS) TORCH_CHECK(false, "torchvulkan [ERROR]: Vulkan error in Cache");

class VulkanCache {
public:
    void softClearCache();
    void clearCache();
    ~VulkanCache() { clearCache(); }

    VkCommandBuffer allocateCommandBuffer(VkCommandPool commandPool);
    void deleteCommandBuffer(VkCommandBuffer commandBuffer);

    VkFence allocateFence();
    void deleteFence(VkFence fence);

    VulkanBuffer* allocateBuffer(size_t size, MemoryUsage usage);
    void deleteBuffer(VulkanBuffer* buffer, MemoryUsage usage);

    void setDevice(VkDevice device) { device_ = device; }
    void setDeviceTable(VolkDeviceTable& table) { device_table = table; }

private:
    VkDevice device_;
    VolkDeviceTable device_table;
    std::mutex mutex_;
    std::vector<VkCommandBuffer> commandBufferPool;
    std::vector<VkFence> fencePool;

    static constexpr size_t NUM_BINS = 64;
    std::vector<VulkanBuffer*> deviceBufferPool[NUM_BINS];
    std::vector<VulkanBuffer*> stagingBufferPool[NUM_BINS];
    size_t getBinIndex(size_t size);
    size_t nextPowerOf2(size_t n);
};