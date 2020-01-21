from dataclasses import dataclass
from typing import Union, Tuple, Iterator, Sized, Container, Any, TypeVar, Generic
from typing import Optional, List, Dict, Sequence, Iterable, Callable
import numpy
import sys

# Use typing_extensions for Python versions < 3.8
if sys.version_info < (3, 8):
    from typing_extensions import Literal, Protocol
else:
    from typing import Literal, Protocol

try:
    import cupy

    get_array_module = cupy.get_array_module
except ImportError:
    get_array_module = lambda obj: numpy


Xp = Union["numpy", "cupy"]  # type: ignore
Shape = Tuple[int, ...]
DTypes = Literal["f", "i", "float32", "int32", "int64", "uint32", "uint64"]
DTypesFloat = Literal["f", "float32"]
DTypesInt = Literal["i", "int32", "int64", "uint32", "uint64"]
OpsNames = Literal["numpy", "cupy", "jax"]
DeviceTypes = Literal["cpu", "gpu", "tpu"]
ArrayT = TypeVar("ArrayT", bound="Array")
Reduced_OutT = TypeVar("Reduced_OutT")


class Array(Generic[ArrayT], Sized, Container):
    T: ArrayT
    base: Optional[ArrayT]

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
    ) -> ArrayT:
        ...

    def copy(self, order: str = ...) -> ArrayT:
        ...

    def fill(self, value: Any) -> None:
        ...

    # Shape manipulation
    def reshape(self: ArrayT, shape: Shape, *, order: str = ...) -> ArrayT:
        ...

    def transpose(self, axes: Shape) -> ArrayT:
        ...

    def flatten(self, order: str = ...) -> ArrayT:
        ...

    def ravel(self, order: str = ...) -> ArrayT:
        ...

    def squeeze(self, axis: Union[int, Shape] = ...) -> ArrayT:
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

    def __copy__(self: ArrayT, order: str = ...) -> ArrayT:
        ...

    def __deepcopy__(self: ArrayT, memo: dict) -> ArrayT:
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

    def __neg__(self: ArrayT) -> ArrayT:
        ...

    def __pos__(self: ArrayT) -> ArrayT:
        ...

    def __abs__(self: ArrayT) -> ArrayT:
        ...

    def __invert__(self: ArrayT) -> ArrayT:
        ...

    def get(self) -> ArrayT:
        ...

    def all(
        self, axis: int = -1, out: Optional[ArrayT] = None, keepdims: bool = False
    ) -> ArrayT:
        ...

    def any(
        self, axis: int = -1, out: Optional[ArrayT] = None, keepdims: bool = False
    ) -> ArrayT:
        ...

    def argmax(self, axis: int = -1, out: Optional[ArrayT] = None) -> ArrayT:
        ...

    def argmin(self, axis: int = -1, out: Optional[ArrayT] = None) -> ArrayT:
        ...

    def clip(self, a_min: Any, a_max: Any, out: Optional[ArrayT]) -> ArrayT:
        ...

    def cumsum(
        self,
        axis: int = -1,
        dtype: Optional[DTypes] = None,
        out: Optional[ArrayT] = None,
    ) -> ArrayT:
        ...

    def max(self, axis: int = -1, out: Optional[ArrayT] = None) -> ArrayT:
        ...

    def mean(
        self,
        axis: int = -1,
        dtype: Optional[DTypes] = None,
        out: Optional[ArrayT] = None,
        keepdims: bool = False,
    ) -> ArrayT:
        ...

    def min(self, axis: int = -1, out: Optional[ArrayT] = None) -> ArrayT:
        ...

    def nonzero(self) -> ArrayT:
        ...

    def prod(
        self,
        axis: int = -1,
        dtype: Optional[DTypes] = None,
        out: Optional[ArrayT] = None,
        keepdims: bool = False,
    ) -> ArrayT:
        ...

    def round(self, decimals: int = 0, out: Optional[ArrayT] = None) -> ArrayT:
        ...

    def sum(
        self,
        axis: int = -1,
        dtype: Optional[DTypes] = None,
        out: Optional[ArrayT] = None,
        keepdims: bool = False,
    ) -> ArrayT:
        ...

    def tobytes(self, order: str = "C") -> bytes:
        ...

    def tolist(self) -> List[Any]:
        ...

    def var(
        self,
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
    @property
    def ptr(self):
        ...

    def get(self) -> NumpyArray:
        ...

    def toDlpack(self) -> "CupyArray":
        ...


def validate_array(obj):
    xp = get_array_module(obj)
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
    if expected_dtype is not None and obj.dtype != expected_dtype:
        xp = get_array_module(obj)
        err = f"wrong array data type (expected {xp.dtype(expected_dtype)}, got {obj.dtype})"
        raise ValueError(err)
    return obj


def get_array_validators(*, ndim=None, dtype=None):
    return (
        lambda v: validate_array_dims(v, ndim),
        lambda v: validate_array_dtype(v, dtype),
    )


class Array1d(Array):
    """1-dimensional array."""

    @classmethod
    def __get_validators__(cls):
        for validator in get_array_validators(ndim=1):
            yield validator


class Array2d(Array):
    """2-dimensional array."""

    @classmethod
    def __get_validators__(cls):
        for validator in get_array_validators(ndim=2):
            yield validator


class Array3d(Array):
    """3-dimensional array."""

    @classmethod
    def __get_validators__(cls):
        for validator in get_array_validators(ndim=3):
            yield validator


class Array4d(Array):
    """4-dimensional array."""

    @classmethod
    def __get_validators__(cls):
        for validator in get_array_validators(ndim=4):
            yield validator


class ArrayNd(Array):
    """N-dimensional array."""

    @classmethod
    def __get_validators__(cls):
        for validator in get_array_validators(ndim=None):
            yield validator


# Union of all int/float array types
ArrayTypes = Union[Array1d, Array2d, Array3d, Array4d, ArrayNd]


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


class Doc(Sized, Container):
    T: "Doc"
    base: Optional["Doc"]

    @property
    def doc(self) -> "Doc":
        ...

    @property
    def start(self) -> int:
        ...

    @property
    def end(self) -> int:
        ...

    def to_array(self, attr_ids: Union[str, int, List[Union[str, int]]]) -> Array:
        ...


InFunc = TypeVar("InFunc")
Wrapper = Callable[[InFunc], InFunc]


class Decorator(Protocol):
    """Protocol to mark a function as returning its child with identical signature."""

    def __call__(self, name: str) -> Callable[[InFunc], InFunc]:
        ...
