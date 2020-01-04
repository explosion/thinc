from typing import Union, Tuple, Callable, Iterator
from enum import Enum


Array = Union["numpy.ndarray", "cupy.ndarray"]  # type: ignore
Xp = Union["numpy", "cupy"]  # type: ignore


class Generator(Iterator):
    """Generator of floats."""

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not hasattr(v, "__iter__") and not hasattr(v, "__next__"):
            raise TypeError("not a valid iterator")
        return v

Shape = Union[Tuple[int], Tuple[int, int], Tuple[int, int, int]]


class NlpType:
    # TODO:
    vocab: "spacy.vocab.Vocab"  # type: ignore  # noqa: F821
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
