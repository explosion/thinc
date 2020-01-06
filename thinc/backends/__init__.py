from typing import Union
import contextlib

from .ops import Ops
from .cupy_ops import CupyOps
from .numpy_ops import NumpyOps
from ..util import create_thread_local


STATE = create_thread_local({"Ops": NumpyOps, "ops": NumpyOps()})


def get_ops(ops: Union[int, str]) -> Union[NumpyOps, CupyOps]:
    if ops in ("numpy", "cpu") or (isinstance(ops, int) and ops < 0):
        return NumpyOps
    elif ops in ("cupy", "gpu") or (isinstance(ops, int) and ops >= 0):
        return CupyOps
    else:
        raise ValueError(f"Invalid ops (or device) description: {ops}")


@contextlib.contextmanager
def use_device(device: Union[str, int]):
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
    "Ops",
    "CupyOps",
    "NumpyOps",
]
