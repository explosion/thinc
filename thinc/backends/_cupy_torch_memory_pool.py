try:
    import torch
    has_torch = True
except ImportError:
    has_torch = False


try:
    import cupy
    import cupy.cuda.memory
    import cupy.cuda.device
    has_cupy = True
except ImportError:
    has_cupy = False


class PyTorchMemory(cupy.cuda.memory.BaseMemory):
    def __init__(self, size_in_bytes: int):
        self._obj = torch.FloatStorage(size_in_bytes // 4)
        self.size = size_in_bytes
    
    @property
    def ptr(self) -> int:
        return self._obj.data_ptr()

    @property
    def device_id(self) -> int:
        return self._obj.device

    @property
    def device(self):
        return cupy.cuda.device.Device(self.device_id)


def cupy_pytorch_allocator(size_in_bytes: int):
    # The pointer object will hold a reference to the `PyTorchMemory` object,
    # which will hold a reference to the torch.FloatStorage object. When
    # Python is ready to free the MemoryPointer, we'll also free the FloatStorage,
    # releasing the memory back to PyTorch.
    return cupy.cuda.memory.MemoryPointer(PyTorchMemory(size_in_bytes), 0)
