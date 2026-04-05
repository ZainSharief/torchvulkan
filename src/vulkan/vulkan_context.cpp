#define VOLK_IMPLEMENTATION
#include "vulkan_context.h"
#include <iostream>

thread_local c10::DeviceIndex VulkanContext::currentDeviceIndex;

VulkanContext& VulkanContext::Instance() 
{
    static VulkanContext vulkanContextInstance;
    return vulkanContextInstance;
}

VulkanContext::VulkanContext()
{
    initVulkan();
    createDeviceContexts();
    createDeviceWithExtensions();
    createDeviceAllocator();
    createDeviceCommandPools();
    validateDevices();
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
    appInfo.apiVersion = VK_API_VERSION_1_3; // if changed, edit VmaAllocatorCreateInfo below

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

void VulkanContext::createDeviceWithExtensions()
{
    float priority = 1.0f;
    
    for (DeviceContext* device : devices) 
    { 
        // chain of feature structs to query what the device supports
        VkPhysicalDeviceFeatures2 supportedFeatures2{};
        supportedFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        VkPhysicalDeviceVulkan11Features supported11{};
        supported11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
        supportedFeatures2.pNext = &supported11;
        VkPhysicalDeviceVulkan12Features supported12{};
        supported12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        supported11.pNext = &supported12;
        VkPhysicalDeviceShaderIntegerDotProductFeatures supportedDotProduct{};
        supportedDotProduct.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES;
        supported12.pNext = &supportedDotProduct;
        VkPhysicalDeviceShaderAtomicFloatFeaturesEXT supportedAtomicFloat{};
        supportedAtomicFloat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;
        supportedDotProduct.pNext = &supportedAtomicFloat;

        // query whch ones are supported
        vkGetPhysicalDeviceFeatures2(device->physicalDevice, &supportedFeatures2);

        // chain of feature structs to enable the features we want (only the ones supported by the device)
        VkPhysicalDeviceShaderAtomicFloatFeaturesEXT enableAtomicFloat{};
        enableAtomicFloat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;
        enableAtomicFloat.shaderBufferFloat32Atomics = supportedAtomicFloat.shaderBufferFloat32Atomics;
        enableAtomicFloat.shaderBufferFloat32AtomicAdd = supportedAtomicFloat.shaderBufferFloat32AtomicAdd;
        enableAtomicFloat.pNext = nullptr; 
        VkPhysicalDeviceShaderIntegerDotProductFeatures enableDotProduct{};
        enableDotProduct.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES;
        enableDotProduct.shaderIntegerDotProduct = supportedDotProduct.shaderIntegerDotProduct;
        enableDotProduct.pNext = &enableAtomicFloat;
        VkPhysicalDeviceVulkan12Features enable12{};
        enable12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        enable12.shaderFloat16 = supported12.shaderFloat16; 
        enable12.shaderInt8 = supported12.shaderInt8;
        enable12.storageBuffer8BitAccess = supported12.storageBuffer8BitAccess;
        enable12.scalarBlockLayout = supported12.scalarBlockLayout;
        enable12.bufferDeviceAddress = supported12.bufferDeviceAddress;
        enable12.pNext = &enableDotProduct;
        VkPhysicalDeviceVulkan11Features enable11{};
        enable11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
        enable11.storageBuffer16BitAccess = supported11.storageBuffer16BitAccess;
        enable11.pNext = &enable12;
        VkPhysicalDeviceFeatures enable10{};
        enable10.shaderFloat64 = supportedFeatures2.features.shaderFloat64;
        enable10.shaderInt64 = supportedFeatures2.features.shaderInt64;
        enable10.shaderInt16 = supportedFeatures2.features.shaderInt16;

        // creating the device 
        VkDeviceQueueCreateInfo queueInfo{};
        queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueInfo.queueFamilyIndex = device->computeQueueFamily;
        queueInfo.queueCount = 1;
        queueInfo.pQueuePriorities = &priority;

        VkPhysicalDeviceFeatures2 features2{};
        features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        features2.features = enable10;
        features2.pNext = &enable11;

        std::vector<const char*> deviceExtensions;
        deviceExtensions.push_back(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);
        deviceExtensions.push_back(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);

        VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        createInfo.queueCreateInfoCount = 1;
        createInfo.pQueueCreateInfos = &queueInfo;
        createInfo.pNext = &features2;
        createInfo.pEnabledFeatures = VK_NULL_HANDLE;
        createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
        createInfo.ppEnabledExtensionNames = deviceExtensions.data();

        if (vkCreateDevice(device->physicalDevice, &createInfo, nullptr, &device->device) == VK_SUCCESS) {
            volkLoadDeviceTable(&device->device_table, device->device);
            device->cache.setDevice(device->device);
            device->cache.setDeviceTable(device->device_table);
            device->device_table.vkGetDeviceQueue(device->device, device->computeQueueFamily, 0, &device->computeQueue);
            continue;
        }
            
        TORCH_WARN(c10::str("torchvulkan [WARNING]: Failed to create a logical device for ", device->properties.deviceName));
        device->valid = false;
    }
}
    
void VulkanContext::createDeviceAllocator()
{
    for (DeviceContext* device : devices) 
    {
        if (!device->valid) continue;

        VmaAllocatorCreateInfo allocatorInfo = {};
        allocatorInfo.physicalDevice = device->physicalDevice;
        allocatorInfo.device = device->device;
        allocatorInfo.instance = instance;
        allocatorInfo.vulkanApiVersion = VK_API_VERSION_1_3;
        allocatorInfo.flags = VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;

        VmaVulkanFunctions vmaFunctions = {};
        vmaFunctions.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
        vmaFunctions.vkGetDeviceProcAddr = vkGetDeviceProcAddr;
        vmaFunctions.vkGetPhysicalDeviceMemoryProperties = vkGetPhysicalDeviceMemoryProperties;
        vmaFunctions.vkGetPhysicalDeviceProperties = vkGetPhysicalDeviceProperties;

        vmaFunctions.vkAllocateMemory = device->device_table.vkAllocateMemory;
        vmaFunctions.vkFreeMemory = device->device_table.vkFreeMemory;
        vmaFunctions.vkMapMemory = device->device_table.vkMapMemory;
        vmaFunctions.vkUnmapMemory = device->device_table.vkUnmapMemory;
        vmaFunctions.vkBindBufferMemory = device->device_table.vkBindBufferMemory;
        vmaFunctions.vkCreateBuffer = device->device_table.vkCreateBuffer;
        vmaFunctions.vkDestroyBuffer = device->device_table.vkDestroyBuffer;
        vmaFunctions.vkGetBufferMemoryRequirements = device->device_table.vkGetBufferMemoryRequirements;
        vmaFunctions.vkFlushMappedMemoryRanges = device->device_table.vkFlushMappedMemoryRanges;
        vmaFunctions.vkInvalidateMappedMemoryRanges = device->device_table.vkInvalidateMappedMemoryRanges;
        vmaFunctions.vkBindBufferMemory2KHR = device->device_table.vkBindBufferMemory2;
        vmaFunctions.vkGetBufferMemoryRequirements2KHR = device->device_table.vkGetBufferMemoryRequirements2;

        allocatorInfo.pVulkanFunctions = &vmaFunctions;

        if (vmaCreateAllocator(&allocatorInfo, &device->allocator) == VK_SUCCESS) continue;
        TORCH_WARN(c10::str("torchvulkan [WARNING]: Failed to create VMA allocator for ", device->properties.deviceName));
        device->valid = false;
    }
}

void VulkanContext::createDeviceCommandPools()
{
    for (DeviceContext* device : devices) 
    {
        if (!device->valid) continue;

        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = device->computeQueueFamily;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

        if (device->device_table.vkCreateCommandPool(device->device, &poolInfo, nullptr, &device->commandPool) == VK_SUCCESS) continue;
        TORCH_WARN(c10::str("torchvulkan [WARNING]: Failed to create a command pool for ", device->properties.deviceName));
        device->valid = false;
    }
}

void VulkanContext::validateDevices()
{
    std::vector<DeviceContext*> validDevices;
    validDevices.reserve(devices.size());

    for (DeviceContext* device : devices) {
        if (device->valid) validDevices.push_back(device);
        else delete device;
    }

    devices = std::move(validDevices);
    if (devices.empty()) TORCH_CHECK(false, "torchvulkan [ERROR]: No devices could be initialized.");
}

VulkanContext::~VulkanContext() 
{
    for (const DeviceContext* device : devices) delete device;
    if (instance != VK_NULL_HANDLE) vkDestroyInstance(instance, nullptr);
}