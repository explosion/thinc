import contextlib
from .base import Ops
from .cupy_ops import CupyOps
from .numpy_ops import NumpyOps
from ..util import create_thread_local, get_ops

STATE = create_thread_local({"Ops": NumpyOps, "ops": NumpyOps()})


@contextlib.contextmanager
def use_device(device):
    """Change the device to execute on for the scope of the block."""
    current_ops = get_current_ops()
    if device == current_ops.device:
        yield
    else:
        set_current_ops(get_ops(device))
        yield
        set_current_ops(current_ops)


def get_current_ops():
    return STATE.ops


def set_current_ops(ops):
    STATE.ops = ops
