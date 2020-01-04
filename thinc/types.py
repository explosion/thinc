from dataclasses import dataclass
from typing import Union, Tuple, Callable, Iterator
from enum import Enum
import numpy


try:
    import cupy

    xp = cupy
except ImportError:
    xp = numpy


Array = Union["numpy.ndarray", "cupy.ndarray"]  # type: ignore
Xp = Union["numpy", "cupy"]  # type: ignore


class Generator(Iterator):
    """Custom generator type. Used to annotate function arguments that accept
    generators so they can be validated by pydantic (which doesn't support
    iterators/iterables otherwise).
    """

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not hasattr(v, "__iter__") and not hasattr(v, "__next__"):
            raise TypeError("not a valid iterator")
        return v


def validate_array(obj):
    if not isinstance(obj, xp.ndarray):
        raise TypeError("not a valid numpy or cupy array")
    return obj


def validate_array_dims(obj, expected_ndim):
    if obj.ndim != expected_ndim:
        err = f"wrong array dimensions (expected {expected_ndim}, got {obj.ndim})"
        raise ValueError(err)
    return obj


def validate_array_dtype(obj, expected_dtype):
    if obj.dtype != expected_dtype:
        err = f"wrong array data type (expected {xp.dtype(expected_dtype)}, got {obj.dtype})"
        raise ValueError(err)
    return obj


def get_array_validators(*, ndim, dtype):
    return (
        lambda v: validate_array(v),
        lambda v: validate_array_dims(v, ndim),
        lambda v: validate_array_dtype(v, dtype),
    )


class Floats1d(xp.ndarray):
    """1-dimensional array of floats."""

    @classmethod
    def __get_validators__(cls):
        for validator in get_array_validators(ndim=1, dtype=xp.float32):
            yield validator


class Floats2d(xp.ndarray):
    """2-dimensional array of floats."""

    @classmethod
    def __get_validators__(cls):
        for validator in get_array_validators(ndim=2, dtype=xp.float32):
            yield validator


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
    np = "numpy"
    cpu = "cpu"
    cp = "cupy"
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
