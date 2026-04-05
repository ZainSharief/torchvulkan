#pragma once
#include <c10/core/Device.h>
#include <volk.h>

struct DeviceContext {
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    uint32_t computeQueueFamily = UINT32_MAX;
    uint32_t vram_heap_index = 0;
    VkPhysicalDeviceProperties properties{};

    ~DeviceContext();
};