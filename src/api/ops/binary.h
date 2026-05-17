#include <iostream>
#include "vulkan/memory.h"
#include "vulkan/vulkan_context.h"
#include "vulkan/allocator.h"
#include "shaders/shader_registry.h"
#include "helpers.h"

#include <c10/core/MemoryFormat.h>
#include <ATen/TensorIterator.h>

enum class BinaryOp {
    ADD = 0,
    SUB = 1,
    RSUB = 2,
    MUL = 3,
    DIV = 4,
    MAX = 5,
    MIN = 6,
    POW = 7,
    RPOW = 8,
    ATAN2 = 9
};

inline torchvulkan::ShaderID get_binaryop_shader_id(at::ScalarType dtype) 
{
    switch (dtype) 
    {
        case at::kDouble: return torchvulkan::ShaderID::BINARYOP_FLOAT64_T_ENTRYPOINT;
        case at::kLong: return torchvulkan::ShaderID::BINARYOP_INT64_T_ENTRYPOINT;
        case at::kUInt64: return torchvulkan::ShaderID::BINARYOP_UINT64_T_ENTRYPOINT;
        
        case at::kFloat: return torchvulkan::ShaderID::BINARYOP_FLOAT32_T_ENTRYPOINT;
        case at::kInt: return torchvulkan::ShaderID::BINARYOP_INT32_T_ENTRYPOINT;
        case at::kUInt32: return torchvulkan::ShaderID::BINARYOP_UINT32_T_ENTRYPOINT;
        
        case at::kHalf: return torchvulkan::ShaderID::BINARYOP_FLOAT16_T_ENTRYPOINT;
        case at::kShort: return torchvulkan::ShaderID::BINARYOP_INT16_T_ENTRYPOINT;
        case at::kUInt16: return torchvulkan::ShaderID::BINARYOP_UINT16_T_ENTRYPOINT;
        
        case at::kChar: return torchvulkan::ShaderID::BINARYOP_INT8_T_ENTRYPOINT;
        case at::kByte: return torchvulkan::ShaderID::BINARYOP_UINT8_T_ENTRYPOINT;
        case at::kBool: return torchvulkan::ShaderID::BINARYOP_UINT8_T_ENTRYPOINT;
        
        default: TORCH_CHECK(false, "torchvulkan [ERROR]: Data type ", c10::toString(dtype), " not supported for binary operations.");
    }
} 

namespace torchvulkan {

template <typename CPUFunc>
at::Tensor binary_op_vulkan(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha, BinaryOp operation, CPUFunc fallback)
{    
    c10::ScalarType promoted_type = at::result_type(self, other);
    
    if (!is_dtype_supported(promoted_type)) {
        TORCH_WARN_ONCE("torchvulkan [WARNING]: Vulkan device does not support ", promoted_type, ". Falling back to CPU.");
        at::Tensor out = fallback(self.cpu(), other.cpu());
        return out.to(self.device());
    }
    
    at::Tensor self_dtype = self.to(promoted_type);
    at::Tensor other_dtype = other.to(self_dtype.options());
    at::Tensor out;

    at::TensorIterator iter = at::TensorIteratorConfig()
        .set_check_mem_overlap(true)
        .add_output(out)
        .add_input(self_dtype)
        .add_input(other_dtype)
        .build();

    out = iter.output();
    uint32_t numel = iter.numel();
    if (numel == 0) return out;

    int32_t out_dims = static_cast<int32_t>(iter.ndim());
    if (out_dims > MAX_DIMS) {
        TORCH_WARN_ONCE("torchvulkan [WARNING]: Coalesced dimensions (", out_dims, ") exceed maximum supported (", MAX_DIMS, "). Falling back to CPU.");
        at::Tensor out = fallback(self.cpu(), other.cpu());
        return out.to(self.device());
    }

    DeviceContext* device = VulkanContext::Instance().CurrentDeviceContext();
    uint32_t vecSize = get_dtype_vec_size(promoted_type);
    uint32_t workgroupSizeX = get_dtype_workgroup_size(promoted_type, vecSize);

    uint32_t contiguous = iter.is_contiguous() ? 1 : 0;
    torchvulkan::ShaderID shader_id = get_binaryop_shader_id(promoted_type);
    uint32_t op = static_cast<uint32_t>(operation);

    SpecializationBuilder spd{};
    spd.push(op)
       .push(contiguous)
       .push(out_dims);
    uint32_t key = (out_dims << 5) | (contiguous << 4) | op;
    SpecializationArgs specialization = {spd.data(), spd.offsets(), spd.sizes(), spd.numConstants(), key};

    IntDivider sizes[MAX_DIMS];
    uint32_t strides_a[MAX_DIMS] = {0};
    uint32_t strides_b[MAX_DIMS] = {0};
    uint32_t strides_out[MAX_DIMS] = {0};
    
    if (!contiguous) {
        int64_t el_size = iter.element_size(0);
        at::IntArrayRef iter_shape = iter.shape();
        at::IntArrayRef iter_strides_out = iter.strides(0);
        at::IntArrayRef iter_strides_a = iter.strides(1);
        at::IntArrayRef iter_strides_b = iter.strides(2);

        for (int i = 0; i < out_dims; i++) {
            sizes[i] = IntDivider(iter_shape[i]);
            strides_a[i] = iter_strides_a[i] / el_size;
            strides_b[i] = iter_strides_b[i] / el_size;
            strides_out[i] = iter_strides_out[i] / el_size;
        }
    }

    PushConstantBuilder pcs{};
    pcs.push(numel)
       .push(alpha.toFloat())
       .push((float)1.0)
       .push((int)0)
       .push_array(sizes)
       .push_array(strides_a)
       .push_array(strides_b)
       .push_array(strides_out);
    
    uint32_t numel_vec = !contiguous ? numel : (numel + (vecSize - 1)) / vecSize;
    uint32_t groupX = (numel_vec + (workgroupSizeX - 1)) / workgroupSizeX;

    VulkanShader shader(shader_id, specialization, device);
    shader.dispatch(
        &pcs, 
        pcs.size(), 
        {self_dtype, other_dtype, out}, 
        groupX, 1, 1
    );

    return out;
}

template <typename CPUFunc>
at::Tensor binary_op_vulkan(const at::Tensor& self, const at::Scalar& other, const at::Scalar& alpha, BinaryOp operation, CPUFunc fallback)
{
    c10::ScalarType promoted_type = at::result_type(self, other);
    
    if (!is_dtype_supported(promoted_type)) {
        TORCH_WARN_ONCE("torchvulkan [WARNING]: Vulkan device does not support ", promoted_type, ". Falling back to CPU.");
        at::Tensor out = fallback(self.cpu(), other);
        return out.to(self.device());
    }
    
    at::Tensor self_dtype = self.to(promoted_type);
    at::Tensor out;

    at::TensorIterator iter = at::TensorIteratorConfig()
        .set_check_mem_overlap(true)
        .add_output(out)
        .add_input(self_dtype)
        .build();

    out = iter.output();
    uint32_t numel = iter.numel();
    if (numel == 0) return out;

    int32_t out_dims = static_cast<int32_t>(iter.ndim());
    if (out_dims > MAX_DIMS) {
        TORCH_WARN_ONCE("torchvulkan [WARNING]: Coalesced dimensions (", out_dims, ") exceed maximum supported (", MAX_DIMS, "). Falling back to CPU.");
        at::Tensor out = fallback(self.cpu(), other);
        return out.to(self.device());
    }
    
    DeviceContext* device = VulkanContext::Instance().CurrentDeviceContext();
    uint32_t vecSize = get_dtype_vec_size(promoted_type);
    uint32_t workgroupSizeX = get_dtype_workgroup_size(promoted_type, vecSize);

    uint32_t contiguous = iter.is_contiguous();
    torchvulkan::ShaderID shader_id = get_binaryop_shader_id(promoted_type);
    uint32_t op = static_cast<uint32_t>(operation);

    SpecializationBuilder spd{};
    spd.push(op)
       .push(contiguous)
       .push(out_dims);
    uint32_t key = (out_dims << 5) | (contiguous << 4) | op;
    SpecializationArgs specialization = {spd.data(), spd.offsets(), spd.sizes(), spd.numConstants(), key};

    IntDivider sizes[MAX_DIMS];
    uint32_t strides_a[MAX_DIMS] = {0};
    uint32_t strides_b[MAX_DIMS] = {0};
    uint32_t strides_out[MAX_DIMS] = {0};

    if (!contiguous) {
        int64_t el_size = iter.element_size(0);
        at::IntArrayRef iter_shape = iter.shape();
        at::IntArrayRef iter_strides_out = iter.strides(0);
        at::IntArrayRef iter_strides_a = iter.strides(1);

        for (int i = 0; i < out_dims; ++i) {
            sizes[i] = IntDivider(iter_shape[i]);
            strides_a[i] = iter_strides_a[i] / el_size;
            strides_out[i] = iter_strides_out[i] / el_size;
        }
    }

    PushConstantBuilder pcs{};
    pcs.push(numel)
       .push(alpha.toFloat())
       .push(other.toFloat())
       .push((int)1)
       .push_array(sizes)
       .push_array(strides_a)
       .push_array(strides_b)
       .push_array(strides_out);

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