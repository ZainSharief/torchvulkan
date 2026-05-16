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
    switch (dtype) 
    {
        case at::kDouble:
        case at::kLong:
        case at::kUInt64: 
            return torchvulkan::ShaderID::COPY_UINT64_T_ENTRYPOINT;
        
        case at::kFloat:
        case at::kInt:
        case at::kUInt32: 
            return torchvulkan::ShaderID::COPY_UINT32_T_ENTRYPOINT;
        
        case at::kHalf:
        case at::kBFloat16: 
        case at::kShort: 
        case at::kUInt16: 
            return torchvulkan::ShaderID::COPY_UINT16_T_ENTRYPOINT;
        
        case at::kChar:
        case at::kByte:
        case at::kBool: 
            return torchvulkan::ShaderID::COPY_UINT8_T_ENTRYPOINT;
        
        default: TORCH_CHECK(false, "torchvulkan [ERROR]: Data type ", c10::toString(dtype), " not supported for copy operations.");
    }
} 