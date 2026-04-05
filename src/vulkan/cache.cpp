#include "cache.h"
#include "vulkan_context.h"

#include <iostream>

VkCommandBuffer VulkanCache::allocateCommandBuffer(VkCommandPool commandPool)
{
    std::unique_lock<std::mutex> lock(mutex_);
    if (!commandBufferPool.empty()) { 
        VkCommandBuffer commandBuffer = commandBufferPool.back();
        commandBufferPool.pop_back();
        VK_CHECK(device_table.vkResetCommandBuffer(commandBuffer, 0));
        return commandBuffer;
    }
    lock.unlock();
    
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer cmd;
    VK_CHECK(device_table.vkAllocateCommandBuffers(device_, &allocInfo, &cmd));
    return cmd;
}

void VulkanCache::deleteCommandBuffer(VkCommandBuffer commandBuffer)
{
    std::lock_guard<std::mutex> lock(mutex_);
    commandBufferPool.push_back(commandBuffer);
}

VkFence VulkanCache::allocateFence()
{
    std::unique_lock<std::mutex> lock(mutex_);
    if (!fencePool.empty()) {
        VkFence fence = fencePool.back();
        fencePool.pop_back();
        VK_CHECK(device_table.vkResetFences(device_, 1, &fence));
        return fence;
    }
    lock.unlock();

    VkFence fence;
    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = 0;
    VK_CHECK(device_table.vkCreateFence(device_, &fenceInfo, nullptr, &fence));
    return fence;
}

void VulkanCache::deleteFence(VkFence fence)
{
    std::lock_guard<std::mutex> lock(mutex_);
    fencePool.push_back(fence);
}

size_t VulkanCache::getBinIndex(size_t size)
{
    if (size == 0) return 0;
    return 64 - __builtin_clzll(size - 1); // builtin compiler method (count 0's)
}

size_t VulkanCache::nextPowerOf2(size_t n) 
{
    if (n <= 1) return 0; 
    
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n |= (n >> 16) >> 16; 
    n++;
    
    return n;
}

VulkanBuffer* VulkanCache::allocateBuffer(size_t size, MemoryUsage usage)
{
    std::vector<VulkanBuffer*>* pools = (usage == MemoryUsage::DEVICE_ONLY) ? deviceBufferPool : stagingBufferPool;
    size_t binIndex = getBinIndex(nextPowerOf2(size));
    std::lock_guard<std::mutex> lock(mutex_);

    if (!pools[binIndex].empty()) {
        VulkanBuffer* buffer = pools[binIndex].back();
        pools[binIndex].pop_back();
        return buffer;
    }

    size_t nextBin = binIndex + 1;
    if (nextBin < NUM_BINS && !pools[nextBin].empty()) {
        VulkanBuffer* buffer = pools[nextBin].back();
        pools[nextBin].pop_back();
        return buffer;
    }

    return VK_NULL_HANDLE;
}

void VulkanCache::deleteBuffer(VulkanBuffer* buffer, MemoryUsage usage)
{
    std::vector<VulkanBuffer*>* pools = (usage == MemoryUsage::DEVICE_ONLY) ? deviceBufferPool : stagingBufferPool;
    size_t binIndex = getBinIndex(nextPowerOf2(buffer->size()));

    std::lock_guard<std::mutex> lock(mutex_);
    pools[binIndex].push_back(buffer);
}

void VulkanCache::softClearCache()
{
    if (device_ == VK_NULL_HANDLE) return;

    for (VkFence fence : fencePool) device_table.vkDestroyFence(device_, fence, nullptr);
    fencePool.clear();

    for (size_t i = 0; i < NUM_BINS; ++i) {
        for (VulkanBuffer* buffer : deviceBufferPool[i]) {
            delete buffer;
        }
        deviceBufferPool[i].clear();

        for (VulkanBuffer* buffer : stagingBufferPool[i]) {
            delete buffer;
        }
        stagingBufferPool[i].clear();
    }

    commandBufferPool.clear();
}

void VulkanCache::clearCache()
{
    if (device_ == VK_NULL_HANDLE) return;

    softClearCache();
}