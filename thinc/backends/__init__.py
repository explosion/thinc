import contextlib
import threading

from .ops import Ops
from .cupy_ops import CupyOps, has_cupy
from .numpy_ops import NumpyOps
from .jax_ops import JaxOps, has_jax, jax_jit
from ._cupy_allocators import cupy_tensorflow_allocator, cupy_pytorch_allocator
from ._param_server import ParamServer
from ..util import create_thread_local
from ..util import assert_tensorflow_installed, assert_pytorch_installed
from ..types import OpsNames


GLOBAL_STATE = {"Ops": NumpyOps, "ops": NumpyOps()}


def get_thread_state():
    """Get a thread-specific state variable that inherits from a global
    state when it's created."""
    thread: threading.Thread = threading.current_thread()
    if not hasattr(thread, "__local"):
        thread.__local = create_thread_local(GLOBAL_STATE)
    return thread.__local


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


def get_ops(name: OpsNames, **kwargs) -> Ops:
    """Get a backend object."""
    ops = {"numpy": NumpyOps, "cupy": CupyOps, "jax": JaxOps}
    if name not in ops:
        raise ValueError(f"Invalid backend: {name}")
    cls = ops[name]
    return cls(**kwargs)


@contextlib.contextmanager
def use_ops(name: OpsNames, **kwargs):
    """Change the backend to execute on for the scope of the block."""
    current_ops = get_current_ops()
    set_current_ops(get_ops(name, **kwargs))
    yield
    set_current_ops(current_ops)


def get_current_ops() -> Ops:
    """Get the current backend object."""
    return get_thread_state().ops


def set_current_ops(ops: Ops) -> None:
    """Change the current backend object."""
    get_thread_state().ops = ops


__all__ = [
    "set_current_ops",
    "get_current_ops",
    "use_ops",
    "jax_jit",
    "ParamServer",
    "Ops",
    "CupyOps",
    "NumpyOps",
    "JaxOps",
    "has_jax",
    "has_cupy",
]
