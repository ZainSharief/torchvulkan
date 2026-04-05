#include <torch/extension.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/macros/Macros.h>

#include "vulkan/vulkan_context.h"

namespace c10 {
namespace impl {

class VulkanGuardImpl : public DeviceGuardImplInterface {
public:
    VulkanGuardImpl() {}
    DeviceType type() const override { return DeviceType::PrivateUse1; }
    
    // we only have 1 compute queue per GPU right now, possibly update in future
    Stream getStream(Device d) const override { return Stream(Stream::UNSAFE, d, 0); }
    Stream exchangeStream(Stream s) const override { return Stream(Stream::UNSAFE, s.device(), 0); }

    Device exchangeDevice(Device d) const override {  
        Device oldDevice = getDevice();
        setDevice(d);
        return oldDevice;
    }

    void setDevice(c10::Device d) const override { 
        if (d.index() >= VulkanContext::Instance().getDeviceCount()) TORCH_CHECK(false, "torchvulkan [ERROR]: Invalid device index");
        VulkanContext::SetCurrentDevice(d.index()); 
    }
    void uncheckedSetDevice(Device d) const noexcept override { VulkanContext::SetCurrentDevice(d.index()); }
    
    c10::Device getDevice() const override { return c10::Device(c10::DeviceType::PrivateUse1, VulkanContext::CurrentDevice()); }
    DeviceIndex deviceCount() const noexcept override { return VulkanContext::Instance().getDeviceCount(); }
};

C10_REGISTER_GUARD_IMPL(PrivateUse1, VulkanGuardImpl);

}
}