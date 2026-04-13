import pytest
import operator
import torch
import torchvulkan as torchvk
from torch.testing import assert_close

DTYPES = [
    torch.float64,
    torch.float32,
    torch.float16,
    torch.int8,
]

SHAPES = [
    (),           # 0D Scalar Tensor
    (0,),         # 1D Empty Tensor
    (7,),         # 1D Prime (Tests shader VEC_SIZE remainder logic)
    (15, 13),     # 2D Odd shapes
    (2, 3, 4, 5), # 4D Batched
]

# Standard binary ops
OPS_TENSOR_TENSOR = [
    (torch.add, "add"),
    (torch.sub, "sub"),
    (torch.mul, "mul"),
    (torch.div, "div"),
    (torch.maximum, "maximum"),
    (torch.minimum, "minimum"),
    (torch.pow, "pow"),
    (torch.atan2, "atan2")
]

def _generate_tensor(shape, dtype):
    """Generates strictly positive values to avoid NaN/Inf in pow/div/atan2"""
    if dtype == torch.int8:
        return torch.randint(1, 10, shape, dtype=dtype)
    return torch.rand(shape, dtype=torch.float32).to(dtype) + 0.1

@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("op, name", OPS_TENSOR_TENSOR)
def test_binary_op_tensor_tensor(shape, dtype, op, name):
    if dtype == torch.int8 and name in ["pow", "atan2"]:
        pytest.skip("Shader does not support pow/atan2 for integers")

    cpu_a = _generate_tensor(shape, dtype)
    cpu_b = _generate_tensor(shape, dtype)
    
    vk_a = cpu_a.to('vulkan')
    vk_b = cpu_b.to('vulkan')
    
    kwargs = {'alpha': 2} if name in ['add', 'sub'] else {}

    expected = op(cpu_a, cpu_b, **kwargs)
    result = op(vk_a, vk_b, **kwargs)
    
    assert result.device.type == 'vulkan'
    assert_close(result.to('cpu'), expected, check_dtype=True)

@pytest.mark.parametrize("op, name", OPS_TENSOR_TENSOR)
def test_binary_op_broadcasting(op, name):
    shapes = [
        ((10, 1), (1, 10)),           # 2D to 2D
        ((5, 1, 4), (1, 3, 4)),       # 3D to 3D
        ((15,), (1, 15)),             # 1D to 2D
        ((), (7, 7)),                 # 0D to 2D
        ((2, 1, 5, 1), (1, 3, 1, 6))  # 4D to 4D
    ]
    
    for shape_a, shape_b in shapes:
        cpu_a = _generate_tensor(shape_a, torch.float32)
        cpu_b = _generate_tensor(shape_b, torch.float32)
        
        vk_a = cpu_a.to('vulkan')
        vk_b = cpu_b.to('vulkan')
        
        expected = op(cpu_a, cpu_b)
        result = op(vk_a, vk_b)
        
        assert_close(result.to('cpu'), expected)

@pytest.mark.parametrize("op, name", OPS_TENSOR_TENSOR)
def test_binary_op_non_contiguous(op, name):
    cpu_a = _generate_tensor((15, 15), torch.float32)
    cpu_b = _generate_tensor((15, 15), torch.float32)
    
    cpu_a_slice = cpu_a[:, ::2] 
    cpu_b_slice = cpu_b[:, ::2] 
    
    vk_a_slice = cpu_a.to('vulkan')[:, ::2]
    vk_b_slice = cpu_b.to('vulkan')[:, ::2]
    
    assert_close(op(vk_a_slice, vk_b_slice).to('cpu'), op(cpu_a_slice, cpu_b_slice))

    cpu_a_trans = cpu_a.t()
    cpu_b_trans = cpu_b.t()
    
    vk_a_trans = cpu_a.to('vulkan').t()
    vk_b_trans = cpu_b.to('vulkan').t()
    
    assert_close(op(vk_a_trans, vk_b_trans).to('cpu'), op(cpu_a_trans, cpu_b_trans))

def test_max_dims_exception():
    """
    Tests the TORCH_CHECK for MAX_DIMS in binary.cpp
    Assuming MAX_DIMS is usually 5 or 8, a 10D tensor should definitely fail.
    """
    shape = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1) # 10D Tensor
    cpu_a = torch.rand(shape)
    cpu_b = torch.rand(shape)
    
    vk_a = cpu_a.to('vulkan')
    vk_b = cpu_b.to('vulkan')
    
    with pytest.raises(RuntimeError, match="Broadcasting supported up to"):
        vk_a + vk_b