from typing import Union
import contextlib

from .ops import Ops
from .cupy_ops import CupyOps
from .numpy_ops import NumpyOps
from ._cupy_allocators import cupy_tensorflow_allocator, cupy_pytorch_allocator
from ._param_server import ParamServer
from ..types import DeviceTypes
from ..util import create_thread_local
from ..util import assert_tensorflow_installed, assert_pytorch_installed


STATE = create_thread_local({"Ops": NumpyOps, "ops": NumpyOps()})


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
    cupy.cuda.set_allocator(cupy_pytorch_allocator)


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
    cupy.cuda.set_allocator(cupy_tensorflow_allocator)


def get_ops(ops: DeviceTypes) -> Union[NumpyOps, CupyOps]:
    if ops in ("numpy", "cpu") or (isinstance(ops, int) and ops < 0):
        return NumpyOps
    elif ops in ("cupy", "gpu") or (isinstance(ops, int) and ops >= 0):
        return CupyOps
    else:
        raise ValueError(f"Invalid ops (or device) description: {ops}")


@contextlib.contextmanager
def use_device(device: DeviceTypes):
    """Change the device to execute on for the scope of the block."""
    current_ops = get_current_ops()
    if device == current_ops.device:
        yield
    else:
        set_current_ops(get_ops(device))
        yield
        set_current_ops(current_ops)


def get_current_ops() -> Ops:
    return STATE.ops


def set_current_ops(ops: Ops) -> None:
    STATE.ops = ops


__all__ = [
    "set_current_ops",
    "get_current_ops",
    "use_device",
    "ParamServer",
    "Ops",
    "CupyOps",
    "NumpyOps",
]
