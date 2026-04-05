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
    createDeviceContexts();
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

    volkLoadInstance(instance);
}

void VulkanContext::createDeviceContexts()
{
    uint32_t physicalDeviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, nullptr);
    if (physicalDeviceCount == 0) TORCH_CHECK(false, "torchvulkan [ERROR]: No Vulkan devices found");

    std::vector<VkPhysicalDevice> physicalDevices(physicalDeviceCount);
    vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, physicalDevices.data());

    for (const VkPhysicalDevice& physicalDevice : physicalDevices) 
    {                
        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
        if (queueFamilyCount == 0) continue;
        
        std::vector<VkQueueFamilyProperties> families(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, families.data());
        
        // if there is a dedicated compute core, we pick that, otherwise we pick any compute & graphics core
        int32_t bestQueueFamily = -1;
        for (uint32_t i = 0; i < queueFamilyCount; i++) 
        {
            if (!(families[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) continue;
            bestQueueFamily = i;
            if (!(families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) break;   
        }
        // make sure there exists a compute core
        if (bestQueueFamily == -1) continue;

        DeviceContext* context = new DeviceContext();
        context->physicalDevice = physicalDevice;
        context->computeQueueFamily = bestQueueFamily;
        vkGetPhysicalDeviceProperties(physicalDevice, &context->properties);

        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

        for (uint32_t i = 0; i < memProperties.memoryHeapCount; ++i) 
        {
            if (memProperties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
                context->vram_heap_index = i;
                break;
            }
        }

        devices.push_back(context);
    }

    if (devices.empty()) {
        TORCH_CHECK(false, "torchvulkan [ERROR]: No compute-capable Vulkan devices found.");
    }
}

VulkanContext::~VulkanContext() 
{ 
    if (instance != VK_NULL_HANDLE) vkDestroyInstance(instance, nullptr);
}