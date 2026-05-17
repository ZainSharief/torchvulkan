#include <torch/extension.h>
#include "api/ops/factory.h"

at::Tensor torchvulkan::empty_memory_format_vulkan(
    c10::SymIntArrayRef size,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> /* device */,
    c10::optional<bool> /* pin_memory */,
    c10::optional<at::MemoryFormat> memory_format) 
{
    at::ScalarType dtype_ = dtype.value_or(at::kFloat);
    at::Layout layout_ = layout.value_or(at::kStrided);
    at::MemoryFormat memory_format_ = memory_format.value_or(at::MemoryFormat::Contiguous);

    TORCH_CHECK(layout_ == at::kStrided, "torchvulkan [ERROR]: Only Strided layout is supported");
    
    std::vector<int64_t> concrete_sizes;
    concrete_sizes.reserve(size.size());
    for (const auto& s : size) {
        concrete_sizes.push_back(s.guard_int(__FILE__, __LINE__));
    }
    
    int64_t numel = c10::multiply_integers(concrete_sizes);
    size_t nbytes = numel * c10::elementSize(dtype_);

    std::vector<int64_t> strides = compute_strides(concrete_sizes, memory_format_);

    auto storage_impl = c10::make_intrusive<c10::StorageImpl>(
        c10::StorageImpl::use_byte_size_t(),
        nbytes,
        c10::GetAllocator(c10::DeviceType::PrivateUse1),
        /* resizeable = */ true 
    );
    
    at::Tensor tensor = at::detail::make_tensor<at::TensorImpl>(
        std::move(storage_impl),
        c10::DispatchKey::PrivateUse1,
        c10::scalarTypeToTypeMeta(dtype_)
    );
    
    tensor.unsafeGetTensorImpl()->set_sizes_and_strides(concrete_sizes, strides);

    return tensor;
}

at::Tensor torchvulkan::empty_strided_vulkan(
    c10::SymIntArrayRef size,
    c10::SymIntArrayRef stride,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> /* layout */, // assumed to be strided, so ignored
    c10::optional<at::Device> /* device */,
    c10::optional<bool> /* pin_memory */) 
{
    at::ScalarType dtype_ = dtype.value_or(at::kFloat);
    
    std::vector<int64_t> concrete_sizes;
    std::vector<int64_t> concrete_strides;
    concrete_sizes.reserve(size.size());
    concrete_strides.reserve(stride.size());
    
    for (const auto& s : size) {
        int64_t val = s.guard_int(__FILE__, __LINE__);
        concrete_sizes.push_back(val);
    }
    for (const auto& s : stride) {
        int64_t val = s.guard_int(__FILE__, __LINE__);
        concrete_strides.push_back(val);
    }
    
    size_t nbytes = at::detail::computeStorageNbytes(
        concrete_sizes, 
        concrete_strides, 
        c10::elementSize(dtype_)
    );

    auto storage_impl = c10::make_intrusive<c10::StorageImpl>(
        c10::StorageImpl::use_byte_size_t(),
        nbytes,
        c10::GetAllocator(c10::DeviceType::PrivateUse1),
        /* resizeable = */ true 
    );
    
    at::Tensor tensor = at::detail::make_tensor<at::TensorImpl>(
        std::move(storage_impl),
        c10::DispatchKey::PrivateUse1,
        c10::scalarTypeToTypeMeta(dtype_)
    );
    
    tensor.unsafeGetTensorImpl()->set_sizes_and_strides(concrete_sizes, concrete_strides);

    return tensor;
}

at::Tensor torchvulkan::copy_from_vulkan(
    const at::Tensor& self, 
    const at::Tensor& dst, 
    bool non_blocking) 
{
    TORCH_CHECK(self.sizes() == dst.sizes(), "torchvulkan [ERROR]: Copy sizes mismatch");

    at::Tensor src = self;
    if (self.scalar_type() != dst.scalar_type()) {
        TORCH_WARN_ONCE("torchvulkan [WARNING]: Data types are not the same. Falling back to CPU for conversion.");
        src = self.to(at::kCPU).to(dst.scalar_type());
    }

    c10::DeviceType src_type = src.device().type();
    c10::DeviceType dst_type = dst.device().type();

    if (dst_type == c10::DeviceType::PrivateUse1) VulkanContext::SetCurrentDevice(dst.device().index());
    else if (src_type == c10::DeviceType::PrivateUse1) VulkanContext::SetCurrentDevice(src.device().index());
    DeviceContext* device = VulkanContext::Instance().CurrentDeviceContext();

    if (src_type == at::kCPU && dst_type == c10::DeviceType::PrivateUse1) 
    {
        void* dest_ptr = (void*)dst.storage().data_ptr().get_context();
        uint64_t dest_offset = dst.storage_offset() * dst.itemsize();
        
        if (!src.is_contiguous()) src = src.contiguous();
        if (dst.is_contiguous()) {
            globalVulkanAllocator.copy_host_to_device(dest_ptr, dest_offset, src.data_ptr(), dst.nbytes());
            return dst;
        }

        at::Tensor stagingBuffer = at::empty_like(dst, dst.options().memory_format(at::MemoryFormat::Contiguous));
        void* staging_buffer_ptr = (void*)stagingBuffer.storage().data_ptr().get_context();
        uint64_t staging_buffer_offset = stagingBuffer.storage_offset() * dst.itemsize();

        globalVulkanAllocator.copy_host_to_device(staging_buffer_ptr, staging_buffer_offset, src.data_ptr(), dst.nbytes());
        dispatch_copy_shader(stagingBuffer, dst);
    }
    else if (src_type == c10::DeviceType::PrivateUse1 && dst_type == at::kCPU) 
    {
        if (!src.is_contiguous()) src = src.contiguous();

        void* src_ptr = (void*)src.storage().data_ptr().get_context();
        uint64_t src_offset = src.storage_offset() * src.itemsize();

        if (dst.is_contiguous()) {    
            globalVulkanAllocator.copy_device_to_host(dst.data_ptr(), src_ptr, src_offset, dst.nbytes());
            return dst;
        }

        at::Tensor stagingBuffer = at::empty_like(dst, dst.options().memory_format(at::MemoryFormat::Contiguous));
        globalVulkanAllocator.copy_device_to_host(stagingBuffer.data_ptr(), src_ptr, src_offset, stagingBuffer.nbytes());
        dst.copy_(stagingBuffer, non_blocking);
    } 
    else if (src_type == c10::DeviceType::PrivateUse1 && dst_type == c10::DeviceType::PrivateUse1) 
    {
        if (!src.is_contiguous() || !dst.is_contiguous()) {
            dispatch_copy_shader(src, dst);
            return dst;
        }

        void* src_ptr = (void*)src.storage().data_ptr().get_context();
        uint64_t src_offset = src.storage_offset() * src.itemsize();
        void* dest_ptr = (void*)dst.storage().data_ptr().get_context();
        uint64_t dest_offset = dst.storage_offset() * dst.itemsize();
        globalVulkanAllocator.copy_device_to_device(dest_ptr, dest_offset, src_ptr, src_offset, dst.nbytes());  
    } 
    else {
        // should never get here, but just in case
        dst.copy_(src.to(at::kCPU), non_blocking);
    }

    return dst;
}

at::Tensor torchvulkan::as_strided_vulkan(
    const at::Tensor& self,
    c10::SymIntArrayRef size,
    c10::SymIntArrayRef stride,
    c10::optional<c10::SymInt> storage_offset)
{
    std::vector<int64_t> concrete_sizes;
    std::vector<int64_t> concrete_strides;
    concrete_sizes.reserve(size.size());
    concrete_strides.reserve(stride.size());
    
    for (const auto& s : size) {
        concrete_sizes.push_back(s.guard_int(__FILE__, __LINE__));
    }
    for (const auto& s : stride) {
        concrete_strides.push_back(s.guard_int(__FILE__, __LINE__));
    }
    
    int64_t offset = storage_offset.has_value() 
        ? storage_offset.value().guard_int(__FILE__, __LINE__) 
        : self.storage_offset();

    at::Tensor result = at::detail::make_tensor<at::TensorImpl>(
        c10::TensorImpl::VIEW,
        c10::Storage(self.storage()),
        self.key_set(),
        self.dtype()
    );
    
    // update the sizes, strides, and offset in the view
    result.unsafeGetTensorImpl()->set_sizes_and_strides(concrete_sizes, concrete_strides);
    result.unsafeGetTensorImpl()->set_storage_offset(offset);
    
    return result;
}

const at::Tensor& torchvulkan::resize_vulkan(
    const at::Tensor& self, 
    c10::IntArrayRef size, 
    c10::optional<at::MemoryFormat> memory_format) 
{
    at::TensorImpl* impl = self.unsafeGetTensorImpl();

    int64_t numel = 1;
    for (auto s : size) numel *= s;
    size_t new_bytes = numel * self.itemsize();

    if (impl->storage().nbytes() < new_bytes) {
        auto new_storage = c10::make_intrusive<c10::StorageImpl>(
            c10::StorageImpl::use_byte_size_t(),
            new_bytes,
            c10::GetAllocator(c10::DeviceType::PrivateUse1),
            /* resizeable = */ true
        );
        impl->set_storage_keep_dtype(std::move(new_storage));
    }

    at::MemoryFormat format = memory_format.value_or(at::MemoryFormat::Contiguous);
    std::vector<int64_t> strides = compute_strides(size, format);
    impl->set_sizes_and_strides(size, strides);

    return self;
}

at::Tensor torchvulkan::contiguous_vulkan(const at::Tensor& self, at::MemoryFormat memory_format) 
{
    if (self.is_contiguous(memory_format)) return self;
    at::Tensor result = at::empty_like(self, self.options().memory_format(memory_format));

    // call directly rather than through dispatcher to avoid weird fallbacks that trigger infinite recursive loops
    return torchvulkan::copy_from_vulkan(self, result, false); 
}

at::Tensor torchvulkan::clone_vulkan(const at::Tensor& self, c10::optional<at::MemoryFormat> memory_format) 
{
    c10::TensorOptions options = self.options();
    if (memory_format.has_value()) options = options.memory_format(memory_format.value());
    at::Tensor result = at::empty_like(self, options);

    // call directly rather than through dispatcher to avoid weird fallbacks that trigger infinite recursive loops
    return torchvulkan::copy_from_vulkan(self, result, false);
}