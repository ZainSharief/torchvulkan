#pragma once
#include "vulkan/vulkan_context.h"
#include <c10/core/Device.h>
#include <c10/core/MemoryFormat.h>
#include <c10/util/strides.h>
#include <vector>
#include <array>
#include <cstdint>
#include <cstring>
#include <stdexcept>

#define MAX_DIMS 4
#define MAX_VEC_SIZE 4
#define MAX_WORKGROUP_BYTES 1024

class SpecializationBuilder {
public:
    // pushes any value and automatically tracks its size and byte offset
    template <typename T>
    SpecializationBuilder& push(const T& value) {
        offsets_array.push_back(data_buffer.size());
        sizes_array.push_back(sizeof(T));

        const uint8_t* ptr = reinterpret_cast<const uint8_t*>(&value);
        data_buffer.insert(data_buffer.end(), ptr, ptr + sizeof(T));
        
        return *this;
    }

    const void* data() const { return data_buffer.data(); }
    const size_t* offsets() const { return offsets_array.data(); }
    const size_t* sizes() const { return sizes_array.data(); }
    uint32_t numConstants() const { return static_cast<uint32_t>(sizes_array.size()); }
    
private:
    std::vector<uint8_t> data_buffer;
    std::vector<size_t> offsets_array;
    std::vector<size_t> sizes_array;
};

class PushConstantBuilder {
public:
    // push a single value
    template <typename T>
    PushConstantBuilder& push(const T& value) 
    {
        size_t size = sizeof(T);
        TORCH_CHECK(current_size + size <= 128, "torchvulkan [ERROR]: Push constants exceeded 128 bytes!");
        
        std::memcpy(buffer.data() + current_size, &value, size);
        current_size += size;
        return *this;
    }

    // push an array
    template <typename T, size_t N>
    PushConstantBuilder& push_array(const T (&arr)[N]) 
    {
        size_t size = sizeof(T) * N;
        TORCH_CHECK(current_size + size <= 128, "torchvulkan [ERROR]: Push constants exceeded 128 bytes!");
        
        std::memcpy(buffer.data() + current_size, arr, size);
        current_size += size;
        return *this;
    }

    const void* data() const { return buffer.data(); }
    size_t size() const { return current_size; }

private:
    std::array<uint8_t, 128> buffer{}; // vulkan guarantees at least 128 bytes of push constant space on all devices
    size_t current_size = 0;
};

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

inline void fill_strides(
    const at::Tensor& self, const at::Tensor& other, const at::Tensor& out, 
    uint32_t* sizes, float* inv_sizes,
    uint32_t* strides_a, uint32_t* strides_b
) {
    uint32_t out_dims = out.dim();
    TORCH_CHECK(out_dims <= MAX_DIMS, "torchvulkan [ERROR]: Output tensor cannot have more than ", MAX_DIMS, " dimensions.");

    for (uint32_t i = 0; i < out_dims; ++i) {
        sizes[i] = out.size(i);
        strides_a[i] = self.stride(i);
        strides_b[i] = other.stride(i);
        inv_sizes[i] = 1.0f / out.size(i);
    }

    for (uint32_t i = out_dims; i < MAX_DIMS; ++i) {
        sizes[i] = 1;
        strides_a[i] = 0;
        strides_b[i] = 0;
        inv_sizes[i] = 1.0f;
    }
}