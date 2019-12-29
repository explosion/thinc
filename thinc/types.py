from typing import Union
from enum import Enum


Array = Union["numpy.ndarray", "cupy.ndarray"]


class OpNames(str, Enum):
    numpy = "numpy"
    cpu = "cpu"
    cupy = "cupy"
    gpu = "gpu"
