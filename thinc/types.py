from typing import Union
from enum import Enum

# TODO: Write proper types
Array = Union["numpy.ndarray", "cupy.ndarray"]


class OpNames(str, Enum):
    numpy = "numpy"
    cpu = "cpu"
    cupy = "cupy"
    gpu = "gpu"
