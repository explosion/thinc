from typing import Union, Tuple, Callable
from enum import Enum


Array = Union["numpy.ndarray", "cupy.ndarray"]  # type: ignore
Xp = Union["numpy", "cupy"]  # type: ignore

Shape = Union[Tuple[int], Tuple[int, int], Tuple[int, int, int]]


class NlpType:
    # TODO:
    vocab: "spacy.vocab.Vocab"  # type: ignore
    pass


class DocType:
    # TODO:
    # DocType = "spacy.tokens.Doc"  # type: ignore
    doc: "DocType"
    to_array: Callable
    start: int
    end: int


class OpNames(str, Enum):
    numpy = "numpy"
    cpu = "cpu"
    cupy = "cupy"
    gpu = "gpu"
