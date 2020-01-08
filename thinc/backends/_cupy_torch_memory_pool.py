from ..util import torch2xp

try:
    import torch
    has_torch = True
except ImportError:
    has_torch = False


def cupy_pytorch_allocator(size_in_bytes: int):
    torch_tensor = torch.zeros((size_in_bytes // 4,))
    cupy_tensor = torch2xp(torch_tensor)
    return cupy_tensor.data
