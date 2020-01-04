from dataclasses import dataclass
from typing import Union, Tuple, Callable, TypeVar

from enum import Enum


Array = Union["numpy.ndarray", "cupy.ndarray"]  # type: ignore
Xp = Union["numpy", "cupy"]  # type: ignore

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


@dataclass
class Ragged:
    data: Array
    lengths: Array


@dataclass
class Padded:
    """A batch of padded sequences, sorted by decreasing length. The data array
    is of shape (step, batch, ...). The auxiliary array size_at_t indicates the
    length of the batch at each timestep, so you can do data[:, :size_at_t[t]] to
    shrink the batch. 
    """
    data: Array
    size_at_t: Array
