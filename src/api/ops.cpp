#include <torch/extension.h>
#include "api/ops/factory.h"

using namespace torchvulkan;

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    // factory
    m.impl("empty.memory_format", &empty_memory_format_vulkan);
    m.impl("empty_strided", &empty_strided_vulkan);
    m.impl("_copy_from", &copy_from_vulkan);
    m.impl("as_strided", &as_strided_vulkan);
}