#pragma once
#include "vulkan/vulkan_context.h"
#include <c10/core/Device.h>
#include <c10/core/MemoryFormat.h>
#include <c10/util/strides.h>
#include <vector>

inline std::vector<int64_t> compute_strides(
    c10::IntArrayRef sizes, 
    at::MemoryFormat format) 
{
    if (format == at::MemoryFormat::Contiguous) {
        auto strides = c10::contiguous_strides(sizes);
        return std::vector<int64_t>(strides.begin(), strides.end());
    }

    if (format == at::MemoryFormat::ChannelsLast && sizes.size() == 4) {
        auto strides = c10::get_channels_last_strides_2d(sizes);
        return std::vector<int64_t>(strides.begin(), strides.end());
    }

    if (format == at::MemoryFormat::ChannelsLast3d && sizes.size() == 5) {
        auto strides = c10::get_channels_last_strides_3d(sizes);
        return std::vector<int64_t>(strides.begin(), strides.end());
    }

    TORCH_CHECK(false, "torchvulkan [ERROR]: Unsupported memory format for sizes.");
}

inline bool is_dtype_supported(at::ScalarType dtype) 
{
    DeviceContext* device = VulkanContext::Instance().CurrentDeviceContext();

    switch (dtype) 
    {
        case at::kDouble: return device->support_float64;
        case at::kLong: return device->support_int64;
        case at::kUInt64: return device->support_int64;
        
        case at::kFloat: return device->support_float32;
        case at::kInt: return device->support_int32;
        case at::kUInt32: return device->support_int32;
        
        case at::kHalf: return device->support_float16;
        case at::kBFloat16: return false; // we cannot support it yet, but when glslc updates we will!
        case at::kShort: return device->support_int16;
        case at::kUInt16: return device->support_int16;
        
        case at::kChar: return device->support_int8; 
        case at::kByte: return device->support_int8; 
        case at::kBool: return device->support_int8; 
        
        default: return false;
    }
}