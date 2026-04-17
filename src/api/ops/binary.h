#include <iostream>
#include "vulkan/memory.h"
#include "vulkan/vulkan_context.h"
#include "vulkan/allocator.h"
#include "shaders/shader_registry.h"
#include "helpers.h"

#include <c10/core/MemoryFormat.h>

inline torchvulkan::ShaderID get_binaryop_shader_id(bool is_contiguous, at::ScalarType dtype, uint32_t& vecSize, uint32_t& workgroupSizeX) 
{
    size_t size = c10::elementSize(dtype);
    vecSize = std::min(static_cast<uint32_t>(16 / size), static_cast<uint32_t>(MAX_VEC_SIZE)); // Vulkan max is vec4
    workgroupSizeX = MAX_WORKGROUP_BYTES / (vecSize * size);

    switch (dtype) {
        case at::kDouble: 
            return is_contiguous ? torchvulkan::ShaderID::BINARYOP_CONTIG_FP64 : torchvulkan::ShaderID::BINARYOP_NONCONTIG_FP64;
        case at::kLong:
        case at::kUInt64: 
            return is_contiguous ? torchvulkan::ShaderID::BINARYOP_CONTIG_INT64 : torchvulkan::ShaderID::BINARYOP_NONCONTIG_INT64;
        case at::kFloat:  
            return is_contiguous ? torchvulkan::ShaderID::BINARYOP_CONTIG_FP32 : torchvulkan::ShaderID::BINARYOP_NONCONTIG_FP32;
        case at::kInt:
        case at::kUInt32: 
            return is_contiguous ? torchvulkan::ShaderID::BINARYOP_CONTIG_INT32 : torchvulkan::ShaderID::BINARYOP_NONCONTIG_INT32;
        case at::kHalf:   
            return is_contiguous ? torchvulkan::ShaderID::BINARYOP_CONTIG_FP16 : torchvulkan::ShaderID::BINARYOP_NONCONTIG_FP16;
        case at::kShort:
        case at::kUInt16: 
            return is_contiguous ? torchvulkan::ShaderID::BINARYOP_CONTIG_INT16 : torchvulkan::ShaderID::BINARYOP_NONCONTIG_INT16;
        case at::kChar:   
        case at::kByte:   
        case at::kBool:   
            return is_contiguous ? torchvulkan::ShaderID::BINARYOP_CONTIG_INT8 : torchvulkan::ShaderID::BINARYOP_NONCONTIG_INT8; 
        default: 
            TORCH_CHECK(false, "torchvulkan [ERROR]: Unsupported scalar type ", c10::toString(dtype), " for binary op.");
    }
}

namespace torchvulkan {

template <typename CPUFunc>
at::Tensor binary_op_vulkan(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha, int32_t operation, CPUFunc fallback)
{    
    c10::ScalarType promoted_type = at::result_type(self, other);
    
    if (!is_dtype_supported(promoted_type)) {
        TORCH_WARN_ONCE("torchvulkan [WARNING]: Vulkan device does not support ", promoted_type, ". Falling back to CPU.");
        at::Tensor out = fallback(self.cpu(), other.cpu());
        return out.to(self.device());
    }
    
    at::Tensor self_dtype = self.to(promoted_type);
    at::Tensor other_dtype = other.to(self_dtype.options());

    c10::DimVector out_size = at::infer_size_dimvector(self_dtype.sizes(), other_dtype.sizes());
    at::Tensor self_expanded = self_dtype.expand(out_size);
    at::Tensor other_expanded = other_dtype.expand(out_size);
    at::Tensor out = at::empty(out_size, self_expanded.options());

    uint32_t numel = out.numel();
    if (numel == 0) return out;

    TORCH_CHECK(out.dim() <= MAX_DIMS, "torchvulkan [ERROR]: Broadcasting supported up to ", MAX_DIMS, " dimensions.");

    uint32_t contiguous = self_expanded.is_contiguous() && other_expanded.is_contiguous();
    DeviceContext* device = VulkanContext::Instance().CurrentDeviceContext();
    uint32_t vecSize, workgroupSizeX;
    torchvulkan::ShaderID shader_id = get_binaryop_shader_id(contiguous, self_expanded.scalar_type(), vecSize, workgroupSizeX);

    int32_t out_dims = static_cast<int32_t>(out.dim());
    int32_t use_scalar = 0;
    SpecializationBuilder spd{};
    spd.push(operation)
       .push(use_scalar)
       .push(out_dims);
    uint32_t key = (out_dims << 5) | (use_scalar << 4) | operation;
    SpecializationArgs specialization = {spd.data(), spd.offsets(), spd.sizes(), spd.numConstants(), key};

    uint32_t sizes[MAX_DIMS];
    uint32_t strides_a[MAX_DIMS];
    uint32_t strides_b[MAX_DIMS];
    float inv_sizes[MAX_DIMS];
    if (!contiguous) fill_strides(self_expanded, other_expanded, out, sizes, inv_sizes, strides_a, strides_b);

    PushConstantBuilder pcs{};
    pcs.push(numel)
       .push(alpha.toFloat())
       .push(1.0f)
       .push_array(sizes)
       .push_array(strides_a)
       .push_array(strides_b)
       .push_array(inv_sizes);
    
    uint32_t numel_vec = !contiguous ? numel : (numel + (vecSize - 1)) / vecSize;
    uint32_t groupX = (numel_vec + (workgroupSizeX - 1)) / workgroupSizeX;

    VulkanShader shader(shader_id, specialization, device);
    shader.dispatch(
        &pcs, 
        pcs.size(), 
        {self_expanded, other_expanded, out}, 
        groupX, 1, 1
    );

    return out;
}

template <typename CPUFunc>
at::Tensor binary_op_vulkan(const at::Tensor& self, const at::Scalar& other, const at::Scalar& alpha, int32_t operation, CPUFunc fallback)
{
    c10::ScalarType promoted_type = at::result_type(self, other);
    
    if (!is_dtype_supported(promoted_type)) {
        TORCH_WARN_ONCE("torchvulkan [WARNING]: Vulkan device does not support ", promoted_type, ". Falling back to CPU.");
        at::Tensor out = fallback(self.cpu(), other);
        return out.to(self.device());
    }
    
    at::Tensor self_dtype = self.to(promoted_type);
    at::Tensor out = at::empty_like(self, self_dtype.options().memory_format(at::MemoryFormat::Contiguous));

    uint32_t numel = out.numel();
    if (numel == 0) return out;

    TORCH_CHECK(out.dim() <= MAX_DIMS, "torchvulkan [ERROR]: Broadcasting supported up to ", MAX_DIMS, " dimensions.");
    
    uint32_t contiguous = self_dtype.is_contiguous();
    DeviceContext* device = VulkanContext::Instance().CurrentDeviceContext();
    uint32_t vecSize, workgroupSizeX;
    torchvulkan::ShaderID shader_id = get_binaryop_shader_id(contiguous, promoted_type, vecSize, workgroupSizeX);

    int32_t out_dims = static_cast<int32_t>(out.dim());
    int32_t use_scalar = 1;
    SpecializationBuilder spd{};
    spd.push(operation)
       .push(use_scalar)
       .push(out_dims);
    uint32_t key = (out_dims << 5) | (use_scalar << 4) | operation;
    SpecializationArgs specialization = {spd.data(), spd.offsets(), spd.sizes(), spd.numConstants(), key};

    uint32_t sizes[MAX_DIMS];
    uint32_t strides_a[MAX_DIMS];
    uint32_t strides_b[MAX_DIMS];
    float inv_sizes[MAX_DIMS];
    if (!contiguous) fill_strides(self_dtype, self_dtype, out, sizes, inv_sizes, strides_a, strides_b);

    PushConstantBuilder pcs{};
    pcs.push(numel)
       .push(alpha.toFloat())
       .push(other.toFloat())
       .push_array(sizes)
       .push_array(strides_a)
       .push_array(strides_b)
       .push_array(inv_sizes);
    
    uint32_t numel_vec = !contiguous ? numel : (numel + (vecSize-1)) / vecSize;
    uint32_t groupX = (numel_vec + (workgroupSizeX - 1)) / workgroupSizeX;
    
    VulkanShader shader(shader_id, specialization, device);
    shader.dispatch(
        &pcs, 
        pcs.size(), 
        {self_dtype, self_dtype, out}, 
        groupX, 1, 1
    );

    return out;
}

at::Tensor add_vulkan(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha);
at::Tensor add_scalar_vulkan(const at::Tensor& self, const at::Scalar& other, const at::Scalar& alpha);
at::Tensor subtract_vulkan(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha);
at::Tensor subtract_scalar_vulkan(const at::Tensor& self, const at::Scalar& other, const at::Scalar& alpha);
at::Tensor rsub_scalar_vulkan(const at::Tensor& self, const at::Scalar& other, const at::Scalar& alpha);
at::Tensor multiply_vulkan(const at::Tensor& self, const at::Tensor& other);
at::Tensor multiply_scalar_vulkan(const at::Tensor& self, const at::Scalar& other);
at::Tensor divide_vulkan(const at::Tensor& self, const at::Tensor& other);
at::Tensor divide_scalar_vulkan(const at::Tensor& self, const at::Scalar& other);
at::Tensor maximum_vulkan(const at::Tensor& self, const at::Tensor& other);
at::Tensor minimum_vulkan(const at::Tensor& self, const at::Tensor& other);
at::Tensor pow_vulkan(const at::Tensor& self, const at::Tensor& other);
at::Tensor pow_tensor_scalar_vulkan(const at::Tensor& self, const at::Scalar& other);
at::Tensor pow_scalar_vulkan(const at::Scalar& self, const at::Tensor& other);
at::Tensor atan2_vulkan(const at::Tensor& self, const at::Tensor& other);

} // namespace torchvulkan