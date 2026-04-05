#include "device_context.h"

VkCommandBuffer DeviceContext::getCommandBuffer()
{
    if (cmd != VK_NULL_HANDLE) return cmd;

    cmd = cache.allocateCommandBuffer(commandPool);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    device_table.vkBeginCommandBuffer(cmd, &beginInfo);

    return cmd;
}

void DeviceContext::flush() 
{
    if (cmd == VK_NULL_HANDLE) return;
        
    device_table.vkEndCommandBuffer(cmd);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;

    VkFence fence = cache.allocateFence();

    std::unique_lock<std::mutex> lock(mutex_);
    device_table.vkQueueSubmit(computeQueue, 1, &submitInfo, fence);
    lock.unlock();
    cmd = VK_NULL_HANDLE;

    if (device_table.vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
        TORCH_CHECK(false, "torchvulkan [ERROR]: Wait for fence failed");
    }

    cache.deleteFence(fence);
}

DeviceContext::~DeviceContext() 
{
    if (commandPool != VK_NULL_HANDLE) device_table.vkDestroyCommandPool(device, commandPool, nullptr);
    if (allocator != VK_NULL_HANDLE) vmaDestroyAllocator(allocator);
    if (device != VK_NULL_HANDLE) device_table.vkDestroyDevice(device, nullptr);
}
