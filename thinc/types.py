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
XY_YZ_OutT = TypeVar("XY_YZ_OutT")
XY_XY_OutT = TypeVar("XY_XY_OutT")
Batchable = Union["Pairs", "Ragged", "Padded", "Array", List, Tuple]


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


@dataclass
class SizedGenerator:
    """A generator that has a __len__ and can repeatedly call the generator
    function.
    """

    get_items: Callable[[], Generator]
    length: int

    def __len__(self):
        return self.length

    def __iter__(self):
        yield from self.get_items()


@dataclass
class Padded:
    """A batch of padded sequences, sorted by decreasing length. The data array
    is of shape (step, batch, ...). The auxiliary array size_at_t indicates the
    length of the batch at each timestep, so you can do data[:, :size_at_t[t]] to
    shrink the batch. The lengths array indicates the length of each row b,
    and the indices indicates the original ordering.
    """

    data: Array3d
    size_at_t: Array1d
    lengths: Array1d
    indices: Array1d

    def __len__(self) -> int:
        return self.lengths.shape[0]

    def __getitem__(self, index) -> "Padded":
        if isinstance(index, int):
            # Slice to keep the dimensionality
            return Padded(
                self.data[:, index : index + 1],
                self.lengths[index : index + 1],
                self.lengths[index : index + 1],
                self.indices[index : index + 1],
            )
        elif isinstance(index, slice):
            return Padded(
                self.data[:, index],
                self.lengths[index],
                self.lengths[index],
                self.indices[index],
            )
        else:
            # If we get a sequence of indices, we need to be careful that
            # we maintain the length-sorting, while also keeping the mapping
            # back to the original order correct.
            sorted_index = list(sorted(index))
            return Padded(
                self.data[sorted_index],
                self.size_at_t[sorted_index],
                self.lengths[sorted_index],
                self.indices[index],  # Use original, to maintain order.
            )


@dataclass
class Ragged:
    """A batch of concatenated sequences, that vary in the size of their
    first dimension. Ragged allows variable-length sequence data to be contiguous
    in memory, without padding.

    Indexing into Ragged is just like indexing into the *lengths* array, except
    it returns a Ragged object with the accompanying sequence data. For instance,
    you can write ragged[1:4] to get a Ragged object with sequences 1, 2 and 3.
    """

    data: Array2d
    lengths: Array1d
    _cumsums: Optional[Array1d] = None

    def __len__(self) -> int:
        return self.lengths.shape[0]

    def __getitem__(self, index: Union[int, slice, Array]) -> "Ragged":
        from .util import get_array_module  # prevent circular imports

        if isinstance(index, tuple):
            raise IndexError("Ragged arrays do not support 2d indexing.")
        starts = self._get_starts()
        ends = self._get_ends()
        if isinstance(index, int):
            s = starts[index]
            e = ends[index]
            return Ragged(self.data[s:e], self.lengths[index : index + 1])
        elif isinstance(index, slice):
            lengths = self.lengths[index]
            cumsums = self._get_cumsums()
            start = cumsums[index.start - 1] if index.start >= 1 else 0
            end = start + lengths.sum()
            return Ragged(self.data[start:end], lengths)
        else:
            # There must be a way to do this "properly" :(. Sigh, hate numpy.
            xp = get_array_module(self.data)
            data = xp.vstack([self[int(i)].data for i in index])
            return Ragged(data, self.lengths[index])

    def _get_cumsums(self) -> Array1d:
        if self._cumsums is None:
            self._cumsums = self.lengths.cumsum()
        return self._cumsums

    def _get_starts(self) -> Array1d:
        from .util import get_array_module

        cumsums = self._get_cumsums()
        xp = get_array_module(cumsums)
        zero = xp.array([0], dtype="i")
        return xp.concatenate((zero, cumsums[:-1]))

    def _get_ends(self) -> Array1d:
        return self._get_cumsums()


_P = TypeVar("_P", bound=Sequence)


@dataclass
class Pairs(Generic[_P]):
    """Dataclass for pairs of sequences that allows indexing into the sequences
    while keeping them aligned.
    """

    one: _P
    two: _P

    def __getitem__(self, index) -> "Pairs[_P]":
        return Pairs(self.one[index], self.two[index])

    def __len__(self) -> int:
        return len(self.one)


@dataclass
class ArgsKwargs:
    """A tuple of (args, kwargs) that can be spread into some function f:

        f(*args, **kwargs)
    """

    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]

    @classmethod
    def from_items(cls, items: Sequence[Tuple[Union[int, str], Any]]) -> "ArgsKwargs":
        """Create an ArgsKwargs object from a sequence of (key, value) tuples,
        such as produced by argskwargs.items(). Each key should be either a string
        or an integer. Items with int keys are added to the args list, and
        items with string keys are added to the kwargs list. The args list is
        determined by sequence order, not the value of the integer.
        """
        args = []
        kwargs = {}
        for key, value in items:
            if isinstance(key, int):
                args.append(value)
            else:
                kwargs[key] = value
        return cls(args=tuple(args), kwargs=kwargs)

    def keys(self) -> Iterable[Union[int, str]]:
        """Yield indices from self.args, followed by keys from self.kwargs."""
        yield from range(len(self.args))
        yield from self.kwargs.keys()

    def values(self) -> Iterable[Any]:
        """Yield elements of from self.args, followed by values from self.kwargs."""
        yield from self.args
        yield from self.kwargs.values()

    def items(self) -> Iterable[Tuple[Union[int, str], Any]]:
        """Yield enumerate(self.args), followed by self.kwargs.items()"""
        yield from enumerate(self.args)
        yield from self.kwargs.items()
