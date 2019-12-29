from typing import Iterable, Any, Union
import numpy
import itertools

try:
    import cupy
    from cupy import get_array_module
except ImportError:
    cupy = None
    get_array_module = lambda _: numpy

try:
    import torch
except ImportError:
    torch = None


from .types import Array, OpNames


def is_cupy_array(arr: Array) -> bool:
    """Check whether an array is a cupy array"""
    if cupy is None:
        return False
    elif isinstance(arr, cupy.ndarray):
        return True
    else:
        return False


def is_numpy_array(arr: Array) -> bool:
    """Check whether an array is a numpy array"""
    if isinstance(arr, numpy.ndarray):
        return True
    else:
        return False


def get_ops(ops: Union[int, OpNames]) -> "thinc.backends.Ops":
    from ..backends import NumpyOps, CupyOps

    if ops in ("numpy", "cpu") or (isinstance(ops, int) and ops < 0):
        return NumpyOps
    elif ops in ("cupy", "gpu") or (isinstance(ops, int) and ops >= 0):
        return CupyOps
    else:
        raise ValueError(f"Invalid ops (or device) description: {ops}")


def set_active_gpu(gpu_id: int):
    import cupy.cuda.device

    device = cupy.cuda.device.Device(gpu_id)
    device.use()
    try:
        import torch

        torch.cuda.set_device(gpu_id)
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    except ImportError:
        pass
    return device


def prefer_gpu(gpu_id: int = 0) -> bool:
    """Use GPU if it's available. Returns True if so, False otherwise."""
    from .ops import CupyOps

    if CupyOps.xp is None:
        return False
    else:
        require_gpu(gpu_id=gpu_id)
        return True


def require_gpu(gpu_id: int = 0) -> bool:
    from ._classes.model import Model
    from .ops import CupyOps

    if CupyOps.xp is None:
        raise ValueError("GPU is not accessible. Was the library installed correctly?")
    Model.Ops = CupyOps
    Model.ops = CupyOps()
    set_active_gpu(gpu_id)
    return True


def minibatch(items: Iterable[Any], size: int = 8) -> Iterable[Any]:
    """Iterate over batches of items. `size` may be an iterator,
    so that batch-size can vary on each step.
    """
    if isinstance(size, int):
        size_ = itertools.repeat(size)
    else:
        size_ = size
    if hasattr(items, "__len__") and hasattr(items, "__getitem__"):
        i = 0
        while i < len(items):
            batch_size = next(size_)
            yield items[i : i + batch_size]
            i += batch_size
    else:
        items = iter(items)
        while True:
            batch_size = next(size_)
            batch = list(itertools.islice(items, int(batch_size)))
            if len(batch) == 0:
                break
            yield list(batch)


def copy_array(dst: Array, src: Array) -> None:
    if isinstance(dst, numpy.ndarray) and isinstance(src, numpy.ndarray):
        dst[:] = src
    elif is_cupy_array(dst):
        src = cupy.array(src, copy=False)
        cupy.copyto(dst, src)
    else:
        numpy.copyto(dst, src)


def to_categorical(y: Array, nb_classes=None):
    # From keras
    xp = get_array_module(y)
    if xp is cupy:
        y = y.get()
    y = numpy.array(y, dtype="int").ravel()
    if not nb_classes:
        nb_classes = numpy.max(y) + 1
    n = y.shape[0]
    categorical = numpy.zeros((n, nb_classes), dtype="float32")
    categorical[numpy.arange(n), y] = 1
    return xp.asarray(categorical)


def is_ragged(seqs) -> bool:
    if isinstance(seqs, tuple) and len(seqs) == 2:
        if len(seqs[0]) == sum(seqs[1]):
            return True
    return False


def get_width(X: Array, dim: int = -1) -> int:
    """Infer the 'width' of a batch of data, which could be any of:
    * An n-dimensional array: Use the shape
    * A tuple (for a ragged array): Use the shape of the first element.
    * A list of arrays (for sequences): Use the shape of the first element.
    """
    if hasattr(X, "shape") and hasattr(X, "ndim"):
        if len(X.shape) == 0:
            return 0
        elif len(X.shape) == 1:
            return int(X.max()) + 1
        else:
            return X.shape[dim]
    elif isinstance(X, tuple) and len(X) == 2:
        return get_width(X[0], dim=dim)
    elif hasattr(X, "__len__") and hasattr(X, "__getitem__"):
        if len(X) == 0:
            return 0
        else:
            return get_width(X[0], dim=dim)
    else:
        err = "Cannot get width of object: has neither shape nor __getitem__"
        raise ValueError(err)


def xp2torch(xp_tensor):
    """Convert a numpy or cupy tensor to a PyTorch tensor."""
    if hasattr(xp_tensor, "toDlpack"):
        return torch.utils.dlpack.from_dlpack(xp_tensor.toDlpack())
    else:
        return torch.from_numpy(xp_tensor)


def torch2xp(torch_tensor):
    """Convert a torch tensor to a numpy or cupy tensor."""
    if torch_tensor.is_cuda:
        return cupy.fromDlpack(torch.utils.dlpack.to_dlpack(torch_tensor))
    else:
        return torch_tensor.detach().numpy()
