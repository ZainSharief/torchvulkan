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

struct IntDivider {
    uint32_t divisor;
    uint32_t multiplier;
    uint32_t shift_val;
    uint32_t pad;

    IntDivider() : divisor(1), multiplier(1), shift_val(0), pad(0) {}

    IntDivider(uint32_t d) 
        : divisor(d) 
    {        
        for (shift_val = 0; shift_val < 32; shift_val++) {
            if ((1U << shift_val) >= d) break;
        }
        
        uint64_t one = 1;
        uint64_t magic = ((one << 32) * ((one << shift_val) - d)) / d + 1;
        multiplier = static_cast<uint32_t>(magic);
    }
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
        case at::kBFloat16: return false; // we cannot support it yet
        case at::kShort: return device->support_int16;
        case at::kUInt16: return device->support_int16;
        
        case at::kChar: return device->support_int8; 
        case at::kByte: return device->support_int8; 
        case at::kBool: return device->support_int8; 
        
        default: return false;
    }
}

inline int get_dtype_vec_size(at::ScalarType dtype) 
{
    switch (dtype) 
    {
        case at::kDouble: return 2;
        case at::kLong: return 2;
        case at::kUInt64: return 2;
        
        case at::kFloat: return 4;
        case at::kInt: return 4;
        case at::kUInt32: return 4;
        
        case at::kHalf: return 4;
        case at::kBFloat16: return 4;
        case at::kShort: return 4;
        case at::kUInt16: return 4;
        
        case at::kChar: return 4; 
        case at::kByte: return 4; 
        case at::kBool: return 4; 
        
        default: return 0;
    }
}

inline int get_dtype_workgroup_size(at::ScalarType dtype) 
{
    switch (dtype) 
    {
        case at::kDouble: return 64;
        case at::kLong: return 64;
        case at::kUInt64: return 64;
        
        case at::kFloat: return 64;
        case at::kInt: return 64;
        case at::kUInt32: return 64;
        
        case at::kHalf: return 128;
        case at::kBFloat16: return 128;
        case at::kShort: return 128;
        case at::kUInt16: return 128;
        
        case at::kChar: return 256; 
        case at::kByte: return 256; 
        case at::kBool: return 256; 
        
        default: return 0;
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