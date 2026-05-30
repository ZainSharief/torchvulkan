#include <iostream>
#include "vulkan/memory.h"
#include "vulkan/vulkan_context.h"
#include "vulkan/allocator.h"
#include "shaders/shader_registry.h"
#include "helpers.h"

#include <c10/core/MemoryFormat.h>

namespace torchvulkan {

at::Tensor mm_vulkan(
    const at::Tensor& self, 
    const at::Tensor& mat2
);

} // namespace torchvulkan