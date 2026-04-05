#pragma once
#include <c10/core/Device.h>
#include <volk.h>

#include "vk_mem_alloc.h"

struct DeviceContext {
    VkDevice device = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkQueue computeQueue = VK_NULL_HANDLE;
    uint32_t computeQueueFamily = UINT32_MAX;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VmaAllocator allocator = VK_NULL_HANDLE;
    uint32_t vram_heap_index = 0;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VolkDeviceTable device_table;
    VkPhysicalDeviceProperties properties{};
    bool valid = true;

    bool is_valid() { return valid; }
    ~DeviceContext();
};