#pragma once
#include <c10/core/Device.h>
#include <volk.h>
#include <vector>

class VulkanContext {
public:
    // singleton pattern
    static VulkanContext& Instance();
    
    // delete copy constructors
    VulkanContext(const VulkanContext&) = delete;
    VulkanContext& operator=(const VulkanContext&) = delete;
    
    VkInstance instance = VK_NULL_HANDLE;

private:
    VulkanContext();
    ~VulkanContext();
    
    void initVulkan();
};