import pytest
import torch
import torchvulkan as torchvk
from torch.testing import assert_close

SHAPES = [
    (),
    (0,),
    (10,),
    (10, 10),
    (2, 3, 4, 5),
]

DTYPES = [
    torch.float64, torch.uint64, torch.int64,
    torch.float32, torch.uint32, torch.int32,
    torch.float16, torch.bfloat16, torch.uint16, torch.int16,
    torch.uint8, torch.int8,
    torch.bool,
]

def _generate_tensor(shape, dtype):
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, dtype=dtype)
    elif dtype.is_floating_point or dtype.is_complex:
        return torch.randn(shape, dtype=dtype)
    else:
        return torch.randint(0, 100, shape, dtype=dtype)

@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_allocation(shape, dtype):
    x = torch.empty(shape, dtype=dtype, device='vulkan')
    
    assert x.device.type == 'vulkan'
    assert x.shape == shape
    assert x.dtype == dtype

@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_round_trip_copy(shape, dtype):
    cpu_tensor = _generate_tensor(shape, dtype)
    
    vulkan_tensor = cpu_tensor.to('vulkan')
    assert vulkan_tensor.device.type == 'vulkan'
    
    cpu_tensor_back = vulkan_tensor.to('cpu')
    assert cpu_tensor_back.device.type == 'cpu'
    
    assert_close(cpu_tensor, cpu_tensor_back)

@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_device_to_device_copy(shape, dtype):
    cpu_tensor = _generate_tensor(shape, dtype)
    vulkan_tensor_1 = cpu_tensor.to('vulkan')
    
    vulkan_tensor_2 = vulkan_tensor_1.clone()
    assert vulkan_tensor_2.device.type == 'vulkan'
    
    assert_close(vulkan_tensor_1.to('cpu'), vulkan_tensor_2.to('cpu'))

def test_as_strided():
    cpu_tensor = torch.randn(10, 10)
    vulkan_tensor = cpu_tensor.to('vulkan')
    
    vulkan_view = vulkan_tensor[:, ::2] 
    cpu_view = cpu_tensor[:, ::2]
    
    assert vulkan_view.device.type == 'vulkan'
    assert vulkan_view.shape == cpu_view.shape
    assert vulkan_view.stride() == cpu_view.stride()
    
    assert vulkan_view.storage_offset() == cpu_view.storage_offset()

def test_empty_strided():
    size = (2, 3)
    stride = (3, 1)
    
    x = torch.empty_strided(size, stride, dtype=torch.float32, device='vulkan')
    
    assert x.device.type == 'vulkan'
    assert x.shape == size
    assert x.stride() == stride