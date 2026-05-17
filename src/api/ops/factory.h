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

inline void dispatch_copy_shader(const at::Tensor& src, const at::Tensor& dst) 
{
    at::TensorIterator iter = at::TensorIteratorConfig()
        .set_check_mem_overlap(true)
        .add_output(dst)
        .add_input(src)
        .build();

    uint32_t numel = iter.numel();
    if (numel == 0) return;
    int32_t out_dims = static_cast<int32_t>(iter.ndim());
    if (out_dims > MAX_DIMS) {
        TORCH_CHECK(false, "torchvulkan [WARNING]: Coalesced dimensions (", out_dims, ") exceed maximum supported (", MAX_DIMS, "). Falling back to CPU.");
    }

    SpecializationBuilder spd{};
    spd.push(out_dims);
    uint32_t key = out_dims;
    SpecializationArgs specialization = {spd.data(), spd.offsets(), spd.sizes(), spd.numConstants(), key};

    DeviceContext* device = VulkanContext::Instance().CurrentDeviceContext();
    torchvulkan::ShaderID shader_id = get_copy_shader_id(dst.scalar_type());
    uint32_t vecSize = get_dtype_vec_size(dst.scalar_type()); // our workgroup must match the shader workgroup
    uint32_t workgroupSizeX = get_dtype_workgroup_size(dst.scalar_type(), vecSize);

    IntDivider sizes[MAX_DIMS];
    uint32_t strides_in[MAX_DIMS] = {0};
    uint32_t strides_out[MAX_DIMS] = {0};
    
    int64_t el_size = iter.element_size(0);
    at::IntArrayRef iter_shape = iter.shape();
    at::IntArrayRef iter_strides_out = iter.strides(0);
    at::IntArrayRef iter_strides_in = iter.strides(1);

    for (int i = 0; i < out_dims; i++) {
        sizes[i] = IntDivider(iter_shape[i]);
        strides_in[i] = iter_strides_in[i] / el_size;
        strides_out[i] = iter_strides_out[i] / el_size;
    }

    PushConstantBuilder pcs{};
    pcs.push(numel)
        .push_array(sizes)
        .push_array(strides_in)
        .push_array(strides_out);

    uint32_t groupX = (numel + (workgroupSizeX - 1)) / workgroupSizeX;

    VulkanShader shader(shader_id, specialization, device);
    shader.dispatch(
        &pcs, 
        pcs.size(), 
        {src, dst}, 
        groupX, 1, 1
    );
}