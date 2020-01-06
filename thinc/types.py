import sys
from dataclasses import dataclass
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Container,
    Generic,
    Iterator,
    List,
    Optional,
    Sized,
    Tuple,
    TypeVar,
    Union,
)

import numpy

# Use typing_extensions for Python versions < 3.8
if sys.version_info < (3, 8):
    from typing_extensions import Literal
else:
    from typing import Literal


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


ndim = TypeVar("ndim", bound=int)

ArrayT = TypeVar("ArrayT", bound="Array")


class Array(Generic[ndim], Sized, Container):
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
        self: ArrayT,
        dtype: DTypes,
        order: str = ...,
        casting: str = ...,
        subok: bool = ...,
        copy: bool = ...,
    ) -> ArrayT:
        ...

    def copy(self: ArrayT, order: str = ...) -> ArrayT:
        ...

    def fill(self: ArrayT, value: Any) -> None:
        ...

    # Shape manipulation
    def reshape(self: ArrayT, shape: Shape, *, order: str = ...) -> ArrayT:
        ...

    def transpose(self: ArrayT, axes: Shape) -> ArrayT:
        ...

    def flatten(self: ArrayT, order: str = ...) -> ArrayT:
        ...

    def ravel(self: ArrayT, order: str = ...) -> ArrayT:
        ...

    def squeeze(self: ArrayT, axis: Union[int, Shape] = ...) -> ArrayT:
        ...

    def __len__(self: ArrayT) -> int:
        ...

    def __getitem__(self: ArrayT, key) -> Any:
        ...

    def __setitem__(self: ArrayT, key, value):
        ...

    def __iter__(self: ArrayT) -> Any:
        ...

    def __contains__(self: ArrayT, key) -> bool:
        ...

    def __index__(self: ArrayT) -> int:
        ...

    def __int__(self: ArrayT) -> int:
        ...

    def __float__(self: ArrayT) -> float:
        ...

    def __complex__(self: ArrayT) -> complex:
        ...

    def __bool__(self: ArrayT) -> bool:
        ...

    def __bytes__(self: ArrayT) -> bytes:
        ...

    def __str__(self: ArrayT) -> str:
        ...

    def __repr__(self: ArrayT) -> str:
        ...

    def __copy__(self: ArrayT, order: str = ...) -> ArrayT:
        ...

    def __deepcopy__(self: ArrayT, memo: dict) -> ArrayT:
        ...

    def __lt__(self: ArrayT, other):
        ...

    def __le__(self: ArrayT, other):
        ...

    def __eq__(self: ArrayT, other):
        ...

    def __ne__(self: ArrayT, other):
        ...

    def __gt__(self: ArrayT, other):
        ...

    def __ge__(self: ArrayT, other):
        ...

    def __add__(self: ArrayT, other):
        ...

    def __radd__(self: ArrayT, other):
        ...

    def __iadd__(self: ArrayT, other):
        ...

    def __sub__(self: ArrayT, other):
        ...

    def __rsub__(self: ArrayT, other):
        ...

    def __isub__(self: ArrayT, other):
        ...

    def __mul__(self: ArrayT, other):
        ...

    def __rmul__(self: ArrayT, other):
        ...

    def __imul__(self: ArrayT, other):
        ...

    def __truediv__(self: ArrayT, other):
        ...

    def __rtruediv__(self: ArrayT, other):
        ...

    def __itruediv__(self: ArrayT, other):
        ...

    def __floordiv__(self: ArrayT, other):
        ...

    def __rfloordiv__(self: ArrayT, other):
        ...

    def __ifloordiv__(self: ArrayT, other):
        ...

    def __mod__(self: ArrayT, other):
        ...

    def __rmod__(self: ArrayT, other):
        ...

    def __imod__(self: ArrayT, other):
        ...

    def __divmod__(self: ArrayT, other):
        ...

    def __rdivmod__(self: ArrayT, other):
        ...

    # NumPy's __pow__ doesn't handle a third argument
    def __pow__(self: ArrayT, other):
        ...

    def __rpow__(self: ArrayT, other):
        ...

    def __ipow__(self: ArrayT, other):
        ...

    def __lshift__(self: ArrayT, other):
        ...

    def __rlshift__(self: ArrayT, other):
        ...

    def __ilshift__(self: ArrayT, other):
        ...

    def __rshift__(self: ArrayT, other):
        ...

    def __rrshift__(self: ArrayT, other):
        ...

    def __irshift__(self: ArrayT, other):
        ...

    def __and__(self: ArrayT, other):
        ...

    def __rand__(self: ArrayT, other):
        ...

    def __iand__(self: ArrayT, other):
        ...

    def __xor__(self: ArrayT, other):
        ...

    def __rxor__(self: ArrayT, other):
        ...

    def __ixor__(self: ArrayT, other):
        ...

    def __or__(self: ArrayT, other):
        ...

    def __ror__(self: ArrayT, other):
        ...

    def __ior__(self: ArrayT, other):
        ...

    def __matmul__(self: ArrayT, other):
        ...

    def __rmatmul__(self: ArrayT, other):
        ...

    def __neg__(self: ArrayT) -> ArrayT:
        ...

    def __pos__(self: ArrayT) -> ArrayT:
        ...

    def __abs__(self: ArrayT) -> ArrayT:
        ...

    def __invert__(self: ArrayT) -> ArrayT:
        ...

    def get(self: ArrayT) -> ArrayT:
        ...

    def all(
        self: ArrayT,
        axis: int = -1,
        out: Optional[ArrayT] = None,
        keepdims: bool = False,
    ) -> ArrayT:
        ...

    def any(
        self: ArrayT,
        axis: int = -1,
        out: Optional[ArrayT] = None,
        keepdims: bool = False,
    ) -> ArrayT:
        ...

    def argmax(self: ArrayT, axis: int = -1, out: Optional[ArrayT] = None) -> ArrayT:
        ...

    def argmin(self: ArrayT, axis: int = -1, out: Optional[ArrayT] = None) -> ArrayT:
        ...

    def clip(self: ArrayT, a_min: Any, a_max: Any, out: Optional[ArrayT]) -> ArrayT:
        ...

    def cumsum(
        self: ArrayT,
        axis: int = -1,
        dtype: Optional[DTypes] = None,
        out: Optional[ArrayT] = None,
    ) -> ArrayT:
        ...

    def max(self: ArrayT, axis: int = -1, out: Optional[ArrayT] = None) -> ArrayT:
        ...

    def mean(
        self: ArrayT,
        axis: int = -1,
        dtype: Optional[DTypes] = None,
        out: Optional[ArrayT] = None,
        keepdims: bool = False,
    ) -> ArrayT:
        ...

    def min(self: ArrayT, axis: int = -1, out: Optional[ArrayT] = None) -> ArrayT:
        ...

    def nonzero(self: ArrayT) -> ArrayT:
        ...

    def prod(
        self: ArrayT,
        axis: int = -1,
        dtype: Optional[DTypes] = None,
        out: Optional[ArrayT] = None,
        keepdims: bool = False,
    ) -> ArrayT:
        ...

    def round(self: ArrayT, decimals: int = 0, out: Optional[ArrayT] = None) -> ArrayT:
        ...

    def sum(
        self: ArrayT,
        axis: int = -1,
        dtype: Optional[DTypes] = None,
        out: Optional[ArrayT] = None,
        keepdims: bool = False,
    ) -> ArrayT:
        ...

    def tobytes(self: ArrayT, order: str = "C") -> bytes:
        ...

    def tolist(self: ArrayT) -> List[Any]:
        ...

    def var(
        self: ArrayT,
        axis: int = -1,
        dtype: Optional[DTypes] = None,
        out: Optional[ArrayT] = None,
        ddof: int = 0,
        keepdims: bool = False,
    ) -> ArrayT:
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
    # TODO: include validate_array here
    if expected_ndim is not None and obj.ndim != expected_ndim:
        err = f"wrong array dimensions (expected {expected_ndim}, got {obj.ndim})"
        raise ValueError(err)
    return obj


def validate_array_dtype(obj, expected_dtype):
    # TODO: include validate_array here
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


f1d1 = TypeVar("f1d1", bound=int)


class Floats1d(Array, Generic[f1d1]):
    """1-dimensional array of floats."""

    @classmethod
    def __get_validators__(cls):
        for validator in get_array_validators(ndim=1, dtype=xp.float32):
            yield validator


f2d1 = TypeVar("f2d1", bound=int)
f2d2 = TypeVar("f2d2", bound=int)


class Floats2d(Generic[f2d1, f2d2], Array[Literal[2]]):
    """2-dimensional array of floats."""

    @classmethod
    def __get_validators__(cls):
        for validator in get_array_validators(ndim=2, dtype=xp.float32):
            yield validator


f3d1 = TypeVar("f3d1", bound=int)
f3d2 = TypeVar("f3d2", bound=int)
f3d3 = TypeVar("f3d3", bound=int)


class Floats3d(Generic[f3d1, f3d2, f3d3], Array[Literal[3]]):
    """3-dimensional array of floats."""

    @classmethod
    def __get_validators__(cls):
        for validator in get_array_validators(ndim=3, dtype=xp.float32):
            yield validator


f4d1 = TypeVar("f4d1", bound=int)
f4d2 = TypeVar("f4d2", bound=int)
f4d3 = TypeVar("f4d3", bound=int)
f4d4 = TypeVar("f4d4", bound=int)


class Floats4d(Generic[f4d1, f4d2, f4d3, f4d4], Array[Literal[4]]):
    """4-dimensional array of floats."""

    @classmethod
    def __get_validators__(cls):
        for validator in get_array_validators(ndim=4, dtype=xp.float32):
            yield validator


f_ndim = TypeVar("f_ndim", bound=int)


class FloatsNd(Array[f_ndim]):
    """N-dimensional array of floats."""

    @classmethod
    def __get_validators__(cls):
        for validator in get_array_validators(ndim=None, dtype=xp.float32):
            yield validator


i1d1 = TypeVar("i1d1", bound=int)


class Ints1d(Generic[i1d1], Array[Literal[1]]):
    """1-dimensional array of ints."""

    @classmethod
    def __get_validators__(cls):
        for validator in get_array_validators(ndim=1, dtype=xp.int32):
            yield validator


i2d1 = TypeVar("i2d1", bound=int)
i2d2 = TypeVar("i2d2", bound=int)


class Ints2d(Generic[i2d1, i2d2], Array[Literal[2]]):
    """2-dimensional array of ints."""

    @classmethod
    def __get_validators__(cls):
        for validator in get_array_validators(ndim=2, dtype=xp.int32):
            yield validator


i3d1 = TypeVar("i3d1", bound=int)
i3d2 = TypeVar("i3d2", bound=int)
i3d3 = TypeVar("i3d3", bound=int)
i3d4 = TypeVar("i3d4", bound=int)


class Ints3d(Generic[i3d1, i3d2, i3d3], Array[Literal[3]]):
    """3-dimensional array of ints."""

    @classmethod
    def __get_validators__(cls):
        for validator in get_array_validators(ndim=3, dtype=xp.int32):
            yield validator


i4d1 = TypeVar("i4d1", bound=int)
i4d2 = TypeVar("i4d2", bound=int)
i4d3 = TypeVar("i4d3", bound=int)
i4d4 = TypeVar("i4d4", bound=int)


class Ints4d(Generic[i4d1, i4d2, i4d3, i4d4], Array[Literal[4]]):
    """4-dimensional array of ints."""

    @classmethod
    def __get_validators__(cls):
        for validator in get_array_validators(ndim=4, dtype=xp.int32):
            yield validator


i_ndim = TypeVar("i_ndim", bound=int)


class IntsNd(Array[i_ndim]):
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


class OpNames(str, Enum):
    np = "numpy"
    cpu = "cpu"
    cp = "cupy"
    gpu = "gpu"


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
