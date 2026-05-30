#include <torch/extension.h>
#include "api/ops/matmul.h"

static const uint32_t TILE_M = 128;
static const uint32_t TILE_N = 128;

at::Tensor torchvulkan::mm_vulkan(
    const at::Tensor& self, 
    const at::Tensor& other)
{
    TORCH_CHECK(self.dim() == 2 && other.dim() == 2, "torchvulkan: mm expects 2D tensors");
    TORCH_CHECK(self.size(1) == other.size(0), "torchvulkan: mat1 and mat2 shapes cannot be multiplied (", self.size(0), "x", self.size(1), " and ", other.size(0), "x", other.size(1), ")");

    uint32_t M = self.size(0);
    uint32_t K = self.size(1);
    uint32_t N = other.size(1);

    c10::ScalarType promoted_type = at::result_type(self, other);
    
    if (!is_dtype_supported(promoted_type)) {
        TORCH_WARN_ONCE("torchvulkan [WARNING]: Vulkan device does not support ", promoted_type, ". Falling back to CPU.");
        return at::mm(self.cpu(), other.cpu()).to(self.device());
    }
    
    at::Tensor self_dtype = self.to(promoted_type).contiguous();
    at::Tensor other_dtype = other.to(self_dtype.options()).contiguous();
    
    at::Tensor out = at::empty({M, N}, self_dtype.options());
    if (M == 0 || N == 0 || K == 0) return out;

    DeviceContext* device = VulkanContext::Instance().CurrentDeviceContext();

    uint32_t workgroupSizeX = 16;
    uint32_t workgroupSizeY = 16;

    torchvulkan::ShaderID shader_id = torchvulkan::get_shader_id_matmul(promoted_type);

    SpecializationBuilder spd{};
    spd.push(workgroupSizeX)
       .push(workgroupSizeY);
    uint32_t key = (workgroupSizeX << 4) | workgroupSizeY;
    SpecializationArgs specialization = {spd.data(), spd.offsets(), spd.sizes(), spd.numConstants(), key};

    PushConstantBuilder pcs{};
    pcs.push(M);
    pcs.push(N);
    pcs.push(K);
    pcs.push((uint32_t)self_dtype.stride(0));  // stride_A (row stride)
    pcs.push((uint32_t)other_dtype.stride(0)); // stride_B (row stride)
    pcs.push((uint32_t)out.stride(0));         // stride_C (row stride)
    pcs.push(1.0f);                            // alpha
    pcs.push(0.0f);                            // beta
    
    uint32_t groupX = (N + TILE_N - 1) / TILE_N;
    uint32_t groupY = (M + TILE_M - 1) / TILE_M;

    VulkanShader shader(shader_id, specialization, device);
    shader.dispatch(
        &pcs, 
        pcs.size(), 
        {self_dtype, other_dtype, out}, 
        groupX, groupY, 1 // <-- 2D Dispatch over X and Y
    );

    return out;
}