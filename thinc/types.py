from typing import Union, Tuple
from enum import Enum


Array = Union["numpy.ndarray", "cupy.ndarray"]
Xp = Union["numpy", "cupy"]

Shape = Tuple[int, int]


class OpNames(str, Enum):
    numpy = "numpy"
    cpu = "cpu"
    cupy = "cupy"
    gpu = "gpu"


class Device(str, Enum):
    cpu = "cpu"
    gpu = "gpu"
