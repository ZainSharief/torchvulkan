try: 
    import torch
except ImportError as e:
    raise ImportError("PyTorch is required to use torchvulkan. Please install PyTorch and try again.") from e

try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"

from . import _C

def is_available() -> bool:
    return _C.is_available()

def device_count() -> int:
    return _C.device_count()

torch.utils.rename_privateuse1_backend('vulkan')
torch._register_device_module('vulkan', object())