#include <iostream>
#include "vulkan/memory.h"
#include "vulkan/vulkan_context.h"
#include "vulkan/allocator.h"
#include "helpers.h"

#include <c10/core/MemoryFormat.h>

namespace torchvulkan {

at::Tensor empty_memory_format_vulkan(
    c10::SymIntArrayRef size,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> /* device */,
    c10::optional<bool> /* pin_memory */,
    c10::optional<at::MemoryFormat> memory_format
);

at::Tensor empty_strided_vulkan(
    c10::SymIntArrayRef size,
    c10::SymIntArrayRef stride,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> /* layout */,
    c10::optional<at::Device> /* device */,
    c10::optional<bool> /* pin_memory */
);

at::Tensor copy_from_vulkan(
    const at::Tensor& self, 
    const at::Tensor& dst, 
    bool non_blocking
); 

at::Tensor as_strided_vulkan(
    const at::Tensor& self,
    c10::SymIntArrayRef size,
    c10::SymIntArrayRef stride,
    c10::optional<c10::SymInt> storage_offset
);

const at::Tensor& resize_vulkan(
    const at::Tensor& self, 
    c10::IntArrayRef size, 
    c10::optional<at::MemoryFormat> memory_format
);

at::Tensor contiguous_vulkan(const at::Tensor& self, at::MemoryFormat memory_format);
at::Tensor clone_vulkan(const at::Tensor& self, c10::optional<at::MemoryFormat> memory_format);

} // namespace torchvulkan

inline torchvulkan::ShaderID get_copy_shader_id(at::ScalarType dtype) 
{
    size_t bytes = c10::elementSize(dtype);
    switch (bytes) 
    {
        case 16:
            return torchvulkan::ShaderID::COPY_16_ENTRYPOINT;
        case 8: 
            return torchvulkan::ShaderID::COPY_8_ENTRYPOINT;
        case 4: 
            return torchvulkan::ShaderID::COPY_4_ENTRYPOINT;
        case 2: 
            return torchvulkan::ShaderID::COPY_2_ENTRYPOINT;
        case 1: 
            return torchvulkan::ShaderID::COPY_1_ENTRYPOINT;
        default: 
            TORCH_CHECK(false, "torchvulkan [ERROR]: Data type ", c10::toString(dtype), " not supported for copy operations.");
    }
} 