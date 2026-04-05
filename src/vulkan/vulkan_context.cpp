#define VOLK_IMPLEMENTATION
#include "vulkan_context.h"
#include <iostream>

VulkanContext& VulkanContext::Instance() 
{
    static VulkanContext vulkanContextInstance;
    return vulkanContextInstance;
}

VulkanContext::VulkanContext()
{
    initVulkan();
}
    
void VulkanContext::initVulkan()
{ 
    volkInitialize();

    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "torchvulkan";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "torchvulkan backend";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    std::vector<const char*> instanceExtensions;

    #ifdef __APPLE__
        instanceExtensions.push_back("VK_KHR_portability_enumeration");
        instanceExtensions.push_back("VK_KHR_get_physical_device_properties2"); 
        createInfo.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    #endif

    createInfo.enabledExtensionCount = static_cast<uint32_t>(instanceExtensions.size());
    createInfo.ppEnabledExtensionNames = instanceExtensions.data();

    if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
        TORCH_CHECK(false, "torchvulkan [ERROR]: Failed to initialize Vulkan.");
    }

    std::cout << "Vulkan Instance Created Successfully." << std::endl;

    volkLoadInstance(instance);

    std::cout << "Volk Instance Loaded Successfully." << std::endl;
}

VulkanContext::~VulkanContext() 
{ 
    if (instance != VK_NULL_HANDLE) vkDestroyInstance(instance, nullptr);
}