from dataclasses import dataclass
from typing import (
    Union,
    Tuple,
    Callable,
    Iterator,
    Sized,
    Container,
    Any,
    Optional,
    List,
)
from enum import Enum
import numpy


try:
    import cupy

    xp = cupy
except ImportError:
    xp = numpy


# type: ignore
Xp = Union["numpy", "cupy"]  # type: ignore
Shape = Tuple[int, ...]


class DTypes(str, Enum):
    f = "f"
    i = "i"
    float32 = "float32"
    int32 = "int32"
    int64 = "int64"
    uint32 = "uint32"
    uint64 = "uint64"


class Array(Sized, Container):
    T: "Array"
    base: Optional["Array"]

    @property
    def dtype(self) -> Any:
        ...

    @property
    def data(self) -> memoryview:
        ...

    @property
    def flags(self) -> Any:
        ...

    @property
    def size(self) -> int:
        ...

    @property
    def itemsize(self) -> int:
        ...

    @property
    def nbytes(self) -> int:
        ...

    @property
    def ndim(self) -> int:
        ...

    @property
    def shape(self) -> Shape:
        ...

    @property
    def strides(self) -> Tuple[int, ...]:
        ...

    def astype(
        self,
        dtype: DTypes,
        order: str = ...,
        casting: str = ...,
        subok: bool = ...,
        copy: bool = ...,
    ) -> "Array":
        ...

    def copy(self, order: str = ...) -> "Array":
        ...

    def fill(self, value: Any) -> None:
        ...

    # Shape manipulation
    def reshape(self, shape: Shape, *, order: str = ...) -> "Array":
        ...

    def transpose(self, axes: Shape) -> "Array":
        ...

    def flatten(self, order: str = ...) -> "Array":
        ...

    def ravel(self, order: str = ...) -> "Array":
        ...

    def squeeze(self, axis: Union[int, Shape] = ...) -> "Array":
        ...

    def __len__(self) -> int:
        ...

    def __getitem__(self, key) -> Any:
        ...

    def __setitem__(self, key, value):
        ...

    def __iter__(self) -> Any:
        ...

    def __contains__(self, key) -> bool:
        ...

    def __index__(self) -> int:
        ...

    def __int__(self) -> int:
        ...

    def __float__(self) -> float:
        ...

    def __complex__(self) -> complex:
        ...

    def __bool__(self) -> bool:
        ...

    def __bytes__(self) -> bytes:
        ...

    def __str__(self) -> str:
        ...

    def __repr__(self) -> str:
        ...

    def __copy__(self: "Array", order: str = ...) -> "Array":
        ...

    def __deepcopy__(self: "Array", memo: dict) -> "Array":
        ...

    def __lt__(self, other):
        ...

    def __le__(self, other):
        ...

    def __eq__(self, other):
        ...

    def __ne__(self, other):
        ...

    def __gt__(self, other):
        ...

    def __ge__(self, other):
        ...

    def __add__(self, other):
        ...

    def __radd__(self, other):
        ...

    def __iadd__(self, other):
        ...

    def __sub__(self, other):
        ...

    def __rsub__(self, other):
        ...

    def __isub__(self, other):
        ...

    def __mul__(self, other):
        ...

    def __rmul__(self, other):
        ...

    def __imul__(self, other):
        ...

    def __truediv__(self, other):
        ...

    def __rtruediv__(self, other):
        ...

    def __itruediv__(self, other):
        ...

    def __floordiv__(self, other):
        ...

    def __rfloordiv__(self, other):
        ...

    def __ifloordiv__(self, other):
        ...

    def __mod__(self, other):
        ...

    def __rmod__(self, other):
        ...

    def __imod__(self, other):
        ...

    def __divmod__(self, other):
        ...

    def __rdivmod__(self, other):
        ...

    # NumPy's __pow__ doesn't handle a third argument
    def __pow__(self, other):
        ...

    def __rpow__(self, other):
        ...

    def __ipow__(self, other):
        ...

    def __lshift__(self, other):
        ...

    def __rlshift__(self, other):
        ...

    def __ilshift__(self, other):
        ...

    def __rshift__(self, other):
        ...

    def __rrshift__(self, other):
        ...

    def __irshift__(self, other):
        ...

    def __and__(self, other):
        ...

    def __rand__(self, other):
        ...

    def __iand__(self, other):
        ...

    def __xor__(self, other):
        ...

    def __rxor__(self, other):
        ...

    def __ixor__(self, other):
        ...

    def __or__(self, other):
        ...

    def __ror__(self, other):
        ...

    def __ior__(self, other):
        ...

    def __matmul__(self, other):
        ...

    def __rmatmul__(self, other):
        ...

    def __neg__(self: "Array") -> "Array":
        ...

    def __pos__(self: "Array") -> "Array":
        ...

    def __abs__(self: "Array") -> "Array":
        ...

    def __invert__(self: "Array") -> "Array":
        ...

    def get(self) -> "Array":
        ...

    def all(
        self, axis: int = -1, out: Optional["Array"] = None, keepdims: bool = False
    ) -> "Array":
        ...

    def any(
        self, axis: int = -1, out: Optional["Array"] = None, keepdims: bool = False
    ) -> "Array":
        ...

    def argmax(self, axis: int = -1, out: Optional["Array"] = None) -> "Array":
        ...

    def argmin(self, axis: int = -1, out: Optional["Array"] = None) -> "Array":
        ...

    def clip(self, a_min: Any, a_max: Any, out: Optional["Array"]) -> "Array":
        ...

    def cumsum(
        self,
        axis: int = -1,
        dtype: Optional[DTypes] = None,
        out: Optional["Array"] = None,
    ) -> "Array":
        ...

    def max(self, axis: int = -1, out: Optional["Array"] = None) -> "Array":
        ...

    def mean(
        self,
        axis: int = -1,
        dtype: Optional[DTypes] = None,
        out: Optional["Array"] = None,
        keepdims: bool = False,
    ) -> "Array":
        ...

    def min(self, axis: int = -1, out: Optional["Array"] = None) -> "Array":
        ...

    def nonzero(self) -> "Array":
        ...

    def prod(
        self,
        axis: int = -1,
        dtype: Optional[DTypes] = None,
        out: Optional["Array"] = None,
        keepdims: bool = False,
    ) -> "Array":
        ...

    def round(self, decimals: int = 0, out: Optional["Array"] = None) -> "Array":
        ...

    def sum(
        self,
        axis: int = -1,
        dtype: Optional[DTypes] = None,
        out: Optional["Array"] = None,
        keepdims: bool = False,
    ) -> "Array":
        ...

    def tobytes(self, order: str = "C") -> bytes:
        ...

    def tolist(self) -> List[Any]:
        ...

    def var(
        self,
        axis: int = -1,
        dtype: Optional[DTypes] = None,
        out: Optional["Array"] = None,
        ddof: int = 0,
        keepdims: bool = False,
    ) -> "Array":
        ...


class NumpyArray(Array):
    pass


class CupyArray(Array):
    def get(self) -> NumpyArray:
        ...


def validate_array(obj):
    if not isinstance(obj, xp.ndarray):
        raise TypeError("not a valid numpy or cupy array")
    return obj


def validate_array_dims(obj, expected_ndim):
    obj = validate_array(obj)  # validate her to make sure it's an array
    if expected_ndim is not None and obj.ndim != expected_ndim:
        err = f"wrong array dimensions (expected {expected_ndim}, got {obj.ndim})"
        raise ValueError(err)
    return obj


def validate_array_dtype(obj, expected_dtype):
    obj = validate_array(obj)  # validate her to make sure it's an array
    if obj.dtype != expected_dtype:
        err = f"wrong array data type (expected {xp.dtype(expected_dtype)}, got {obj.dtype})"
        raise ValueError(err)
    return obj


def get_array_validators(*, ndim, dtype):
    return (
        lambda v: validate_array_dims(v, ndim),
        lambda v: validate_array_dtype(v, dtype),
    )


class Floats1d(Array):
    """1-dimensional array of floats."""

    @classmethod
    def __get_validators__(cls):
        for validator in get_array_validators(ndim=1, dtype=xp.float32):
            yield validator


class Floats2d(Array):
    """2-dimensional array of floats."""

    @classmethod
    def __get_validators__(cls):
        for validator in get_array_validators(ndim=2, dtype=xp.float32):
            yield validator


class Floats3d(Array):
    """3-dimensional array of floats."""

    @classmethod
    def __get_validators__(cls):
        for validator in get_array_validators(ndim=3, dtype=xp.float32):
            yield validator


class Floats4d(Array):
    """4-dimensional array of floats."""

    @classmethod
    def __get_validators__(cls):
        for validator in get_array_validators(ndim=4, dtype=xp.float32):
            yield validator


class FloatsNd(Array):
    """N-dimensional array of floats."""

    @classmethod
    def __get_validators__(cls):
        for validator in get_array_validators(ndim=None, dtype=xp.float32):
            yield validator


class Ints1d(Array):
    """1-dimensional array of ints."""

    @classmethod
    def __get_validators__(cls):
        for validator in get_array_validators(ndim=1, dtype=xp.int32):
            yield validator


class Ints2d(Array):
    """2-dimensional array of ints."""

    @classmethod
    def __get_validators__(cls):
        for validator in get_array_validators(ndim=2, dtype=xp.int32):
            yield validator


class Ints3d(Array):
    """3-dimensional array of ints."""

    @classmethod
    def __get_validators__(cls):
        for validator in get_array_validators(ndim=3, dtype=xp.int32):
            yield validator


class Ints4d(Array):
    """4-dimensional array of ints."""

    @classmethod
    def __get_validators__(cls):
        for validator in get_array_validators(ndim=4, dtype=xp.int32):
            yield validator


class IntsNd(Array):
    """N-dimensional array of ints."""

    @classmethod
    def __get_validators__(cls):
        for validator in get_array_validators(ndim=None, dtype=xp.int32):
            yield validator


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


# This should probably become a dataclass too.
RNNState = Tuple[Tuple[Floats2d, Floats2d], Floats2d]


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


__all__ = [
    "Array",
    "Floats1d",
    "Floats2d",
    "Floats3d",
    "Floats4d",
    "FloatsNd",
    "Ints1d",
    "Ints2d",
    "Ints3d",
    "Ints4d",
    "IntsNd",
    "RNNState",
    "Ragged",
    "Padded",
    "Xp",
    "Shape",
    "DTypes",
]
