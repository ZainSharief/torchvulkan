#include <torch/extension.h>
#include "vulkan/vulkan_context.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "torchvulkan backend";

    m.def("is_available", []() {
        return VulkanContext::Instance().instance != nullptr;
    });

    m.def("device_count", []() {
        return VulkanContext::Instance().getDeviceCount();
    });
}