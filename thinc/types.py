from typing import Union, Tuple
from enum import Enum


Array = Union["numpy.ndarray", "cupy.ndarray"]
Xp = Union["numpy", "cupy"]

Shape = Union[Tuple[int], Tuple[int, int], Tuple[int, int, int]]
DocType = "spacy.tokens.Doc"


class OpNames(str, Enum):
    numpy = "numpy"
    cpu = "cpu"
    cupy = "cupy"
    gpu = "gpu"
