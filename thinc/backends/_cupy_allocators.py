from typing import Any, cast
from ..types import CupyArray
from ..util import tensorflow2xp
from ..util import torch2xp

try:
    import tensorflow
except ImportError:
    pass

try:
    import torch
except ImportError:
    pass

try:
    from cupy.cuda.memory import MemoryPointer as MemoryPointerT
except ImportError:
    MemoryPointerT = Any


def cupy_tensorflow_allocator(size_in_bytes: int) -> MemoryPointerT:
    """Function that can be passed into cupy.cuda.set_allocator, to have cupy
    allocate memory via TensorFlow. This is important when using the two libraries
    together, as otherwise OOM errors can occur when there's available memory
    sitting in the other library's pool.
    """
    tensor = tensorflow.zeros((size_in_bytes // 4,), dtype=tensorflow.dtypes.float32)
    # We convert to cupy via dlpack, so that we can get a memory pointer.
    cupy_array = cast(CupyArray, tensorflow2xp(tensor))
    # Now return the array's memory pointer.
    return cupy_array.ptr


def cupy_pytorch_allocator(size_in_bytes: int) -> MemoryPointerT:
    """Function that can be passed into cupy.cuda.set_allocator, to have cupy
    allocate memory via PyTorch. This is important when using the two libraries
    together, as otherwise OOM errors can occur when there's available memory
    sitting in the other library's pool.
    """
    torch_tensor = torch.zeros((size_in_bytes // 4,))
    cupy_tensor = cast(CupyArray, torch2xp(torch_tensor))
    return cupy_tensor.data
