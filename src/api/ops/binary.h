#include <iostream>
#include "vulkan/memory.h"
#include "vulkan/vulkan_context.h"
#include "vulkan/allocator.h"
#include "shaders/shader_registry.h"
#include "helpers.h"

#include <c10/core/MemoryFormat.h>

#define MAX_DIMS 4

struct BinaryOpSpecializationData {
    int32_t op_type; 
    int32_t use_scalar;
    int32_t ndim;

    uint32_t pack() const { return (ndim << 5) | (use_scalar << 4) | op_type; }    
    static constexpr uint32_t numConstants() { return 3; }
    static constexpr const size_t* sizes() { static constexpr size_t s[] = { sizeof(BinaryOpSpecializationData::op_type), sizeof(BinaryOpSpecializationData::use_scalar), sizeof(BinaryOpSpecializationData::ndim) }; return s; }
    static constexpr const size_t* offsets() { static constexpr size_t o[] = { offsetof(BinaryOpSpecializationData, op_type), offsetof(BinaryOpSpecializationData, use_scalar), offsetof(BinaryOpSpecializationData, ndim) }; return o; }
};

struct BinaryOpPushConstants {
    uint32_t numel;
    float alpha;
    float scalar_value;
    uint32_t sizes[MAX_DIMS];
    uint32_t strides_a[MAX_DIMS];
    uint32_t strides_b[MAX_DIMS];
    float inv_sizes[MAX_DIMS];

    inline void fill_strides(const at::Tensor* self, const at::Tensor* other, const at::Tensor* out, const uint32_t out_dims)
    {
        for (uint32_t i = 0; i < out_dims; ++i) {
            sizes[i] = out->size(i);
            strides_a[i] = self->stride(i);
            strides_b[i] = other->stride(i);
            inv_sizes[i] = 1.0f / out->size(i);
        }

        for (uint32_t i = out_dims; i < MAX_DIMS; ++i) {
            sizes[i] = 1;
            strides_a[i] = 0;
            strides_b[i] = 0;
            inv_sizes[i] = 1.0f;
        }
    }
};

inline torchvulkan::ShaderID get_binaryop_shader_id(bool is_contiguous, at::ScalarType dtype, uint32_t* workgroupSizeX) 
{
    *workgroupSizeX = dtype == at::kDouble ? 128 : 64;

    if (is_contiguous) {
        switch (dtype) {
            case at::kDouble: return torchvulkan::ShaderID::BINARYOP_CONTIG_FP64;
            case at::kLong: return torchvulkan::ShaderID::BINARYOP_CONTIG_INT64;
            case at::kUInt64: return torchvulkan::ShaderID::BINARYOP_CONTIG_INT64;
            
            case at::kFloat: return torchvulkan::ShaderID::BINARYOP_CONTIG_FP32;
            case at::kInt: return torchvulkan::ShaderID::BINARYOP_CONTIG_INT32;
            case at::kUInt32: return torchvulkan::ShaderID::BINARYOP_CONTIG_INT32;
            
            case at::kHalf: return torchvulkan::ShaderID::BINARYOP_CONTIG_FP16;
            case at::kShort: return torchvulkan::ShaderID::BINARYOP_CONTIG_INT16;
            case at::kUInt16: return torchvulkan::ShaderID::BINARYOP_CONTIG_INT16;
            
            case at::kChar: return torchvulkan::ShaderID::BINARYOP_CONTIG_INT8; 
            case at::kByte: return torchvulkan::ShaderID::BINARYOP_CONTIG_INT8; 
            case at::kBool: return torchvulkan::ShaderID::BINARYOP_CONTIG_INT8; 
            default: TORCH_CHECK(false, "torchvulkan [ERROR]: Unsupported scalar type ", c10::toString(dtype), "for binary op.");
        }
    } else {
        switch (dtype) {
            case at::kDouble: return torchvulkan::ShaderID::BINARYOP_NONCONTIG_FP64;
            case at::kLong: return torchvulkan::ShaderID::BINARYOP_NONCONTIG_INT64;
            case at::kUInt64: return torchvulkan::ShaderID::BINARYOP_NONCONTIG_INT64;
            
            case at::kFloat: return torchvulkan::ShaderID::BINARYOP_NONCONTIG_FP32;
            case at::kInt: return torchvulkan::ShaderID::BINARYOP_NONCONTIG_INT32;
            case at::kUInt32: return torchvulkan::ShaderID::BINARYOP_NONCONTIG_INT32;
            
            case at::kHalf: return torchvulkan::ShaderID::BINARYOP_NONCONTIG_FP16;
            case at::kShort: return torchvulkan::ShaderID::BINARYOP_NONCONTIG_INT16;
            case at::kUInt16: return torchvulkan::ShaderID::BINARYOP_NONCONTIG_INT16;
            
            case at::kChar: return torchvulkan::ShaderID::BINARYOP_NONCONTIG_INT8; 
            case at::kByte: return torchvulkan::ShaderID::BINARYOP_NONCONTIG_INT8; 
            case at::kBool: return torchvulkan::ShaderID::BINARYOP_NONCONTIG_INT8; 
            default: TORCH_CHECK(false, "torchvulkan [ERROR]: Unsupported scalar type ", c10::toString(dtype), "for binary op.");
        }
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
    uint32_t workgroupSizeX;
    torchvulkan::ShaderID shader_id = get_binaryop_shader_id(contiguous, self_expanded.scalar_type(), &workgroupSizeX);

    int32_t out_dims = static_cast<int32_t>(out.dim());
    BinaryOpSpecializationData spd = { operation, 0, out_dims };
    SpecializationArgs specialization = { (void*)&spd, spd.offsets(), spd.sizes(), spd.numConstants(), spd.pack() };

    VulkanShader shader(shader_id, specialization, device);
    BinaryOpPushConstants pcs = { numel, alpha.toFloat(), 1.0f };
    if (!contiguous) pcs.fill_strides(&self_expanded, &other_expanded, &out, out_dims);

    uint32_t numel_vec4 = !contiguous || self_expanded.scalar_type() == at::ScalarType::Double ? numel : (numel + 3) / 4; // FIX
    uint32_t groupX = (numel_vec4 + (workgroupSizeX - 1)) / workgroupSizeX; // FIX
    size_t push_size = contiguous ? (sizeof(uint32_t) + sizeof(float) + sizeof(float)) : sizeof(BinaryOpPushConstants);

    shader.dispatch(
        &pcs, 
        push_size, 
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
    at::Tensor out = at::empty_like(self, self_dtype.options());

    uint32_t numel = out.numel();
    if (numel == 0) return out;

    TORCH_CHECK(out.dim() <= MAX_DIMS, "torchvulkan [ERROR]: Broadcasting supported up to ", MAX_DIMS, " dimensions.");
    
    uint32_t contiguous = self_dtype.is_contiguous();
    DeviceContext* device = VulkanContext::Instance().CurrentDeviceContext();
    uint32_t workgroupSizeX;
    torchvulkan::ShaderID shader_id = get_binaryop_shader_id(contiguous, promoted_type, &workgroupSizeX);

    int32_t out_dims = static_cast<int32_t>(out.dim());
    BinaryOpSpecializationData spd = { operation, 1, out_dims };
    SpecializationArgs specialization = { (void*)&spd, spd.offsets(), spd.sizes(), spd.numConstants(), spd.pack() };

    VulkanShader shader(shader_id, specialization, device);
    BinaryOpPushConstants pcs = { numel, alpha.toFloat(), other.toFloat() };
    if (!contiguous) pcs.fill_strides(&self_dtype, &self_dtype, &out, out_dims);

    uint32_t numel_vec4 = !contiguous || promoted_type == at::ScalarType::Double ? numel : (numel + 3) / 4;
    uint32_t groupX = (numel_vec4 + (workgroupSizeX - 1)) / workgroupSizeX;
    size_t push_size = contiguous ? (sizeof(uint32_t) + sizeof(float) + sizeof(float)) : sizeof(BinaryOpPushConstants);

    shader.dispatch(
        &pcs, 
        push_size, 
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