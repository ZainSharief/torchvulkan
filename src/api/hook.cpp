#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <c10/core/Device.h>

#include "vulkan/vulkan_context.h"
#include "vulkan/allocator.h" 

struct VulkanHooksInterface : public at::PrivateUse1HooksInterface 
{
    bool isBuilt() const override { return true; }
    bool isAvailable() const override { return VulkanContext::Instance().getDeviceCount() > 0; }
    bool hasPrimaryContext(c10::DeviceIndex device_index) const override { return device_index < VulkanContext::Instance().getDeviceCount(); }
    at::Device getDeviceFromPtr(void* data) const override { return c10::Device(c10::DeviceType::PrivateUse1, 0); }

    c10::Allocator* getPinnedMemoryAllocator() const override {
        return c10::GetAllocator(c10::DeviceType::CPU);
    }

    void resizePrivateUse1Bytes(const c10::Storage& storage, size_t newsize) const override {
        TORCH_CHECK(false, "torchvulkan [ERROR]: Native resizing is not implemented yet.");
    }

    const at::Generator& getDefaultGenerator(c10::DeviceIndex device_index) const override {
        TORCH_CHECK(false, "torchvulkan [ERROR]: getDefaultGenerator not implemented yet.");
    }

    at::Generator getNewGenerator(c10::DeviceIndex device_index = -1) const override {
        TORCH_CHECK(false, "torchvulkan [ERROR]: getNewGenerator not implemented yet.");
    }
};

static VulkanHooksInterface vulkan_hooks;
static bool is_hooks_registered = []() {
    at::RegisterPrivateUse1HooksInterface(&vulkan_hooks);
    return true;
}();