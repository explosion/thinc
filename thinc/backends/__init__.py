import contextlib
from typing import Type, Dict

from contextvars import ContextVar

from .ops import Ops
from .cupy_ops import CupyOps, has_cupy
from .numpy_ops import NumpyOps
from ._cupy_allocators import cupy_tensorflow_allocator, cupy_pytorch_allocator
from ._param_server import ParamServer
from ..util import assert_tensorflow_installed, assert_pytorch_installed
from ..util import is_cupy_array
from ..types import OpsNames


context_ops: ContextVar[NumpyOps] = ContextVar("context_ops", default=NumpyOps())
context_Ops: ContextVar[Type[NumpyOps]] = ContextVar("context_Ops", default=NumpyOps)
context_pools: ContextVar[dict] = ContextVar("context_pools", default={})


def set_gpu_allocator(allocator: str) -> None:  # pragma: no cover
    """Route GPU memory allocation via PyTorch or tensorflow.
    Raise an error if the given argument does not match either of the two.
    """
    if allocator == "pytorch":
        use_pytorch_for_gpu_memory()
    elif allocator == "tensorflow":
        use_tensorflow_for_gpu_memory()
    else:
        raise ValueError(f"Invalid 'gpu_allocator' argument: '{allocator}")


def use_pytorch_for_gpu_memory() -> None:  # pragma: no cover
    """Route GPU memory allocation via PyTorch.

    This is recommended for using PyTorch and cupy together, as otherwise
    OOM errors can occur when there's available memory sitting in the other
    library's pool.

    We'd like to support routing Tensorflow memory allocation via PyTorch as well
    (or vice versa), but do not currently have an implementation for it.
    """
    import cupy.cuda

    assert_pytorch_installed()
    pools = context_pools.get()
    if "pytorch" not in pools:
        pools["pytorch"] = cupy.cuda.MemoryPool(allocator=cupy_pytorch_allocator)
    cupy.cuda.set_allocator(pools["pytorch"].malloc)


def use_tensorflow_for_gpu_memory() -> None:  # pragma: no cover
    """Route GPU memory allocation via TensorFlow.

    This is recommended for using TensorFlow and cupy together, as otherwise
    OOM errors can occur when there's available memory sitting in the other
    library's pool.

    We'd like to support routing PyTorch memory allocation via Tensorflow as
    well (or vice versa), but do not currently have an implementation for it.
    """
    import cupy.cuda

    assert_tensorflow_installed()
    pools = context_pools.get()
    if "tensorflow" not in pools:
        pools["tensorflow"] = cupy.cuda.MemoryPool(allocator=cupy_tensorflow_allocator)
    cupy.cuda.set_allocator(pools["tensorflow"].malloc)


def get_ops(name: OpsNames, **kwargs) -> Ops:
    """Get a backend object."""
    ops = {"numpy": NumpyOps, "cupy": CupyOps}
    if name not in ops:
        raise ValueError(f"Invalid backend: {name}")
    cls = ops[name]
    return cls(**kwargs)


def get_array_ops(arr):
    """Return an Ops object to match the array's device and backend."""
    if is_cupy_array(arr):
        return CupyOps()
    else:
        return NumpyOps()


@contextlib.contextmanager
def use_ops(name: OpsNames, **kwargs):
    """Change the backend to execute on for the scope of the block."""
    current_ops = get_current_ops()
    set_current_ops(get_ops(name, **kwargs))
    yield
    set_current_ops(current_ops)


def get_current_ops() -> Ops:
    """Get the current backend object."""
    return context_ops.get()


def set_current_ops(ops: Ops) -> None:
    """Change the current backend object."""
    context_ops.set(ops)


__all__ = [
    "set_current_ops",
    "get_current_ops",
    "use_ops",
    "ParamServer",
    "Ops",
    "CupyOps",
    "NumpyOps",
    "has_cupy",
]
