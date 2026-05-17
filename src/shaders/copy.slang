#include "helpers/int_divider.slang"

[[vk::constant_id(0)]] const int NDIM = 0;

struct PushConstants
{
    IntDivider sizes[4];
    uint4 strides_in;
    uint4 strides_out;
    uint numel;    
};

[[vk::push_constant]] ConstantBuffer<PushConstants> pcs;

void copyKernel<
    T,
    let ndim : int
>(
    uint3 gi: SV_DispatchThreadID,
    StructuredBuffer<T> tensor_in,
    RWStructuredBuffer<T> tensor_out
)
{
    uint idx = gi.x;
    if (idx >= pcs.numel) return;

    uint idx_in = 0;
    uint idx_out = 0;
    uint linear_idx = idx;

    [unroll]
    for (int i = 0; i < ndim; i++) {
        uint next_linear_idx = fast_div(linear_idx, pcs.sizes[i]);
        uint coord = linear_idx - (next_linear_idx * pcs.sizes[i].divisor);
        linear_idx = next_linear_idx;
        idx_in += coord * pcs.strides_in[i];
        idx_out += coord * pcs.strides_out[i];
    }

    tensor_out[idx_out] = tensor_in[idx_in];
}