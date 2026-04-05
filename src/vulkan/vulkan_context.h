#pragma once
#include <c10/core/Device.h>
#include <volk.h>
#include <vector>

#include "device_context.h"

class VulkanContext {
public:
    // singleton pattern
    static VulkanContext& Instance();
    
    // delete copy constructors
    VulkanContext(const VulkanContext&) = delete;
    VulkanContext& operator=(const VulkanContext&) = delete;
    
    VkInstance instance = VK_NULL_HANDLE;
    std::vector<DeviceContext*> devices;

    uint32_t getDeviceCount() { return devices.size(); }

private:
    VulkanContext();
    ~VulkanContext();
    
    void initVulkan();
    void createDeviceContexts();
};