from typing import Union, Tuple, Sized, Container, Any, TypeVar, Callable
from typing import Iterable, Iterator, Sequence, Dict, Generic, cast
from typing import Optional, List, overload
from dataclasses import dataclass
import numpy
import sys

try:
    import cupy

    get_array_module = cupy.get_array_module
except ImportError:
    get_array_module = lambda obj: numpy

# Use typing_extensions for Python versions < 3.8
if sys.version_info < (3, 8):
    from typing_extensions import Protocol, Literal
else:
    from typing import Protocol, Literal  # noqa: F401


# fmt: off
XY_YZ_OutT = TypeVar("XY_YZ_OutT")
XY_XY_OutT = TypeVar("XY_XY_OutT")

OpsNames = Literal["numpy", "cupy"]
DeviceTypes = Literal["cpu", "gpu", "tpu"]
Batchable = Union["Pairs", "Ragged", "Padded", "ArrayXd", List, Tuple]
Xp = Union["numpy", "cupy"]  # type: ignore
Shape = Tuple[int, ...]
DTypes = Literal["f", "i", "float16", "float32", "float64", "int32", "int64", "uint32", "uint64"]
DTypesFloat = Literal["f", "float32", "float16", "float64"]
DTypesInt = Literal["i", "int32", "int64", "uint32", "uint64"]

Array1d = Union["Floats1d", "Ints1d"]
Array2d = Union["Floats2d", "Ints2d"]
Array3d = Union["Floats3d", "Ints3d"]
Array4d = Union["Floats4d", "Ints4d"]
FloatsXd = Union["Floats1d", "Floats2d", "Floats3d", "Floats4d"]
IntsXd = Union["Ints1d", "Ints2d", "Ints3d", "Ints4d"]
ArrayXd = Union[FloatsXd, IntsXd]
List1d = Union[List["Floats1d"], List["Ints1d"]]
List2d = Union[List["Floats2d"], List["Ints2d"]]
List3d = Union[List["Floats3d"], List["Ints3d"]]
List4d = Union[List["Floats4d"], List["Ints4d"]]
ListXd = Union[List["FloatsXd"], List["IntsXd"]]

ArrayT = TypeVar("ArrayT")
SelfT = TypeVar("SelfT")
Array1dT = TypeVar("Array1dT", bound="Array1d")

# These all behave the same as far as indexing is concerned
Slicish = Union[slice, List[int], "ArrayXd"]
_1_KeyScalar = int
_1_Key1d = Slicish
_1_AllKeys = Union[_1_KeyScalar, _1_Key1d]
_F1_AllReturns = Union[float, "Floats1d"]
_I1_AllReturns = Union[int, "Ints1d"]

_2_KeyScalar = Tuple[int, int]
_2_Key1d = Union[int, Tuple[Slicish, int], Tuple[int, Slicish]]
_2_Key2d = Union[Tuple[Slicish, Slicish], Slicish]
_2_AllKeys = Union[_2_KeyScalar, _2_Key1d, _2_Key2d]
_F2_AllReturns = Union[float, "Floats1d", "Floats2d"]
_I2_AllReturns = Union[int, "Ints1d", "Ints2d"]

_3_KeyScalar = Tuple[int, int, int]
_3_Key1d = Union[Tuple[int, int], Tuple[int, int, Slicish], Tuple[int, Slicish, int], Tuple[Slicish, int, int]]
_3_Key2d = Union[int, Tuple[int, Slicish], Tuple[Slicish, int], Tuple[int, Slicish, Slicish], Tuple[Slicish, int, Slicish], Tuple[Slicish, Slicish, int]]
_3_Key3d = Union[Slicish, Tuple[Slicish, Slicish], Tuple[Slicish, Slicish, Slicish]]
_3_AllKeys = Union[_3_KeyScalar, _3_Key1d, _3_Key2d, _3_Key3d]
_F3_AllReturns = Union[float, "Floats1d", "Floats2d", "Floats3d"]
_I3_AllReturns = Union[int, "Ints1d", "Ints2d", "Ints3d"]

# Typedefs for the reduction methods.
Tru = Literal[True]
Fal = Literal[False]
OneAx = Union[int, Tuple[int]]
TwoAx = Tuple[int, int]
ThreeAx = Tuple[int, int, int]
FourAx = Tuple[int, int, int, int]
_1_AllAx = Optional[OneAx]
_2_AllAx = Union[Optional[TwoAx], OneAx]
_3_AllAx = Union[Optional[ThreeAx], TwoAx, OneAx]
_4_AllAx = Union[Optional[FourAx], ThreeAx, TwoAx, OneAx]
_1F_ReduceResults = Union[float, "Floats1d"]
_2F_ReduceResults = Union[float, "Floats1d", "Floats2d"]
_3F_ReduceResults = Union[float, "Floats1d", "Floats2d", "Floats3d"]
_4F_ReduceResults = Union[float, "Floats1d", "Floats2d", "Floats3d", "Floats4d"]
_1I_ReduceResults = Union[int, "Ints1d"]
_2I_ReduceResults = Union[int, "Ints1d", "Ints2d"]
_3I_ReduceResults = Union[int, "Ints1d", "Ints2d", "Ints3d"]
_4I_ReduceResults = Union[int, "Ints1d", "Ints2d", "Ints3d", "Ints4d"]

# TODO:
# We need to get correct overloads in for the following reduction methods.
# The 'sum' reduction is correct --- the others need to be just the same,
# but with a different name.

# max, min, prod, round, var, mean, ptp, std

# There's also one *slightly* different function, cumsum. This doesn't
# have a scalar version -- it always makes an array.


class _Array(Sized, Container):
    @classmethod
    def __get_validators__(cls):
        """Runtime validation for pydantic."""
        yield lambda v: validate_array(v)

    @property
    def dtype(self) -> DTypes: ...
    @property
    def data(self) -> memoryview: ...
    @property
    def flags(self) -> Any: ...
    @property
    def size(self) -> int: ...
    @property
    def itemsize(self) -> int: ...
    @property
    def nbytes(self) -> int: ...
    @property
    def ndim(self) -> int: ...
    @property
    def shape(self) -> Shape: ...
    @property
    def strides(self) -> Tuple[int, ...]: ...

    # TODO: Is ArrayT right?
    def astype(self: ArrayT, dtype: DTypes, order: str = ..., casting: str = ..., subok: bool = ..., copy: bool = ...) -> ArrayT: ...
    def copy(self: ArrayT, order: str = ...) -> ArrayT: ...
    def fill(self, value: Any) -> None: ...
    # Shape manipulation
    def reshape(self: ArrayT, shape: Shape, *, order: str = ...) -> ArrayT: ...
    def transpose(self: ArrayT, axes: Shape) -> ArrayT: ...
    # TODO: is this right? It returns 1d
    def flatten(self, order: str = ...): ...
    # TODO: is this right? It returns 1d
    def ravel(self, order: str = ...): ...
    def squeeze(self, axis: Union[int, Shape] = ...): ...
    def __len__(self) -> int: ...
    def __setitem__(self, key, value): ...
    def __iter__(self) -> Iterator[Any]: ...
    def __contains__(self, key) -> bool: ...
    def __index__(self) -> int: ...
    def __int__(self) -> int: ...
    def __float__(self) -> float: ...
    def __complex__(self) -> complex: ...
    def __bool__(self) -> bool: ...
    def __bytes__(self) -> bytes: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    def __copy__(self, order: str = ...): ...
    def __deepcopy__(self, memo: dict) -> ArrayT: ...
    def __lt__(self, other): ...
    def __le__(self, other): ...
    def __eq__(self, other): ...
    def __ne__(self, other): ...
    def __gt__(self, other): ...
    def __ge__(self, other): ...
    def __add__(self, other): ...
    def __radd__(self, other): ...
    def __iadd__(self, other): ...
    def __sub__(self, other): ...
    def __rsub__(self, other): ...
    def __isub__(self, other): ...
    def __mul__(self, other): ...
    def __rmul__(self, other): ...
    def __imul__(self, other): ...
    def __truediv__(self, other): ...
    def __rtruediv__(self, other): ...
    def __itruediv__(self, other): ...
    def __floordiv__(self, other): ...
    def __rfloordiv__(self, other): ...
    def __ifloordiv__(self, other): ...
    def __mod__(self, other): ...
    def __rmod__(self, other): ...
    def __imod__(self, other): ...
    def __divmod__(self, other): ...
    def __rdivmod__(self, other): ...
    # NumPy's __pow__ doesn't handle a third argument
    def __pow__(self, other): ...
    def __rpow__(self, other): ...
    def __ipow__(self, other): ...
    def __lshift__(self, other): ...
    def __rlshift__(self, other): ...
    def __ilshift__(self, other): ...
    def __rshift__(self, other): ...
    def __rrshift__(self, other): ...
    def __irshift__(self, other): ...
    def __and__(self, other): ...
    def __rand__(self, other): ...
    def __iand__(self, other): ...
    def __xor__(self, other): ...
    def __rxor__(self, other): ...
    def __ixor__(self, other): ...
    def __or__(self, other): ...
    def __ror__(self, other): ...
    def __ior__(self, other): ...
    def __matmul__(self, other): ...
    def __rmatmul__(self, other): ...
    def __neg__(self: ArrayT) -> ArrayT: ...
    def __pos__(self: ArrayT) -> ArrayT: ...
    def __abs__(self: ArrayT) -> ArrayT: ...
    def __invert__(self: ArrayT) -> ArrayT: ...
    def get(self: ArrayT) -> ArrayT: ...
    def all(self, axis: int = -1, out: Optional[ArrayT] = None, keepdims: bool = False) -> ArrayT: ...
    def any(self, axis: int = -1, out: Optional[ArrayT] = None, keepdims: bool = False) -> ArrayT: ...
    # def argmax(self, axis: int = -1, out: Optional["Array"] = None, keepdims: Union[Tru, Fal]=False) -> Union[int, "Ints1d"]: ...
    def argmin(self, axis: int = -1, out: Optional[ArrayT] = None) -> ArrayT: ...
    def clip(self, a_min: Any, a_max: Any, out: Optional[ArrayT]) -> ArrayT: ...
    #def cumsum( self: ArrayT, axis: int = -1, dtype: Optional[DTypes] = None, out: Optional[ArrayT] = None) -> ArrayT: ...
    def max(self, axis: int = -1, out: Optional[ArrayT] = None) -> ArrayT: ...
    # def mean(self, axis: int = -1, dtype: Optional[DTypes] = None, out: Optional[SelfT] = None, keepdims: bool = False) -> "Array": ...
    def min(self, axis: int = -1, out: Optional[ArrayT] = None) -> ArrayT: ...
    def nonzero(self) -> ArrayT: ...
    def prod(self, axis: int = -1, dtype: Optional[DTypes] = None, out: Optional[ArrayT] = None, keepdims: bool = False) -> ArrayT: ...
    def round(self, decimals: int = 0, out: Optional[ArrayT] = None) -> ArrayT: ...
    # def sum(self, axis: int = -1, dtype: Optional[DTypes] = None, out: Optional[ArrayT] = None, keepdims: bool = False) -> ArrayT: ...
    def tobytes(self, order: str = "C") -> bytes: ...
    def tolist(self) -> List[Any]: ...
    def var(self: SelfT, axis: int = -1, dtype: Optional[DTypes] = None, out: Optional[ArrayT] = None, ddof: int = 0, keepdims: bool = False) -> SelfT: ...


class _Floats(_Array):
    @property
    def dtype(self) -> DTypesFloat: ...

    def fill(self, value: float) -> None: ...
    def reshape(self, shape: Shape, *, order: str = ...) -> "_Floats": ...


class _Ints(_Array):
    @property
    def dtype(self) -> DTypesInt: ...

    def fill(self, value: int) -> None: ...
    def reshape(self, shape: Shape, *, order: str = ...) -> "_Ints": ...


"""
Extensive overloads to represent __getitem__ behaviour.

In an N+1 dimensional array, there will be N possible return types. For instance,
if you have a 2d array, you could get back a float (array[i, j]), a floats1d
(array[i]) or a floats2d (array[:i, :j]). You'll get the scalar if you have N
ints in the index, a 1d array if you have N-1 ints, etc.

So the trick here is to make a union with the various combinations that produce
each result type, and then only have one overload per result. If we overloaded
on each *key* type, that would get crazy, because there's tonnes of combinations.

In each rank, we can use the same key-types for float and int, but we need a
different return-type union.
"""


class _Array1d(_Array):
    """1-dimensional array."""

    @classmethod
    def __get_validators__(cls):
        """Runtime validation for pydantic."""
        yield lambda v: validate_array(v, ndim=1)

    @property
    def ndim(self) -> Literal[1]: ...
    @property
    def shape(self) -> Tuple[int]: ...

    def __iter__(self) -> Iterator[Union[float, int]]: ...
    def astype(self, dtype: DTypes, order: str = ..., casting: str = ..., subok: bool = ..., copy: bool = ...) -> "_Array1d": ...
    def flatten(self: SelfT, order: str = ...) -> SelfT: ...
    def ravel(self: SelfT, order: str = ...) -> SelfT: ...
    # These is actually a bit too strict: It's legal to say 'array1d + array2d'
    # That's kind of bad code though; it's better to write array2d + array1d.
    # We could relax this, but let's try the strict version.
    def __add__(self: SelfT, other: Union[float, int, "Array1d"]) -> SelfT: ...
    def __sub__(self: SelfT, other: Union[float, int, "Array1d"]) -> SelfT: ...
    def __mul__(self: SelfT, other: Union[float, int, "Array1d"]) -> SelfT: ...
    def __pow__(self: SelfT, other: Union[float, int, "Array1d"]) -> SelfT: ...
    def __matmul__(self: SelfT, other: Union[float, int, "Array1d"]) -> SelfT: ...
    # These are not too strict though: you can't do += with higher dimensional.
    def __iadd__(self, other: Union[float, int, "Array1d"]): ...
    def __isub__(self, other: Union[float, int, "Array1d"]): ...
    def __imul__(self, other: Union[float, int, "Array1d"]): ...
    def __ipow__(self, other: Union[float, int, "Array1d"]): ...

    @overload
    def argmax(self, keepdims: Fal = False, axis: int = -1, out: Optional[_Array] = None) -> int: ...
    @overload
    def argmax(self, keepdims: Tru, axis: int = -1, out: Optional[_Array] = None) -> "Ints1d": ...
    def argmax(self, keepdims: bool = False, axis: int = -1, out: Optional[_Array] = None) -> Union[int, "Ints1d"]: ...

    @overload
    def mean(self, keepdims: Tru, axis: int = -1, dtype: Optional[DTypes] = None, out: Optional["Floats1d"] = None) -> "Floats1d": ...
    @overload
    def mean(self, keepdims: Fal = False, axis: int = -1, dtype: Optional[DTypes] = None, out: Optional["Floats1d"] = None) -> float: ...
    def mean(self, keepdims: bool = False, axis: int = -1, dtype: Optional[DTypes] = None, out: Optional["Floats1d"] = None) -> Union["Floats1d", float]: ...


class Floats1d(_Array1d, _Floats):
    """1-dimensional array of floats."""

    T: "Floats1d"

    @classmethod
    def __get_validators__(cls):
        """Runtine validation for pydantic."""
        yield lambda v: validate_array(v, ndim=1, dtype="f")

    def __iter__(self) -> Iterator[float]: ...

    @overload
    def __getitem__(self, key: _1_KeyScalar) -> float: ...
    @overload
    def __getitem__(self, key: _1_Key1d) -> "Floats1d": ...
    def __getitem__(self, key: _1_AllKeys) -> _F1_AllReturns: ...

    @overload
    def __setitem__(self, key: _1_KeyScalar, value: float) -> None: ...
    @overload
    def __setitem__(self, key: _1_Key1d, value: "Floats1d") -> None: ...
    def __setitem__(self, key: _1_AllKeys, _F1_AllReturns) -> None: ...

    @overload
    def cumsum(self, *, keepdims: Tru, axis: Optional[OneAx] = None, out: Optional["Floats1d"] = None) -> "Floats1d": ...
    @overload # Cumsum is unusual in this
    def cumsum(self, *, keepdims: Fal, axis: Optional[OneAx] = None, out: Optional["Floats1d"] = None) -> "Floats1d": ...
    def cumsum(self, *, keepdims: bool = False, axis: _1_AllAx = None, out: Optional["Floats1d"] = None) -> "Floats1d": ...

    @overload
    def sum(self, *, keepdims: Tru, axis: Optional[OneAx] = None, out: Optional["Floats1d"] = None) -> "Floats1d": ...
    @overload
    def sum(self, *, keepdims: Fal, axis: Optional[OneAx] = None, out = None) -> float: ...
    def sum(self, *, keepdims: bool = False, axis: _1_AllAx = None, out: Optional["Floats1d"] = None) -> _1F_ReduceResults: ...


class Ints1d(_Array1d, _Ints):
    """1-dimensional array of ints."""

    T: "Ints1d"

    @classmethod
    def __get_validators__(cls):
        """Runtime validation for pydantic."""
        yield lambda v: validate_array(v, ndim=1, dtype="i")

    def __iter__(self) -> Iterator[int]: ...

    @overload
    def __getitem__(self, key: _1_KeyScalar) -> int: ...
    @overload
    def __getitem__(self, key: _1_Key1d) -> "Ints1d": ...
    def __getitem__(self, key: _1_AllKeys) -> _I1_AllReturns: ...

    @overload
    def __setitem__(self, key: _1_KeyScalar, value: int) -> None: ...
    @overload
    def __setitem__(self, key: _1_Key1d, value: Union[int, "Ints1d"]) -> None: ...
    def __setitem__(self, key: _1_AllKeys, _I1_AllReturns) -> None: ...

    @overload
    def cumsum(self, *, keepdims: Tru, axis: Optional[OneAx] = None, out: Optional["Ints1d"] = None) -> "Ints1d": ...
    @overload
    def cumsum(self, *, keepdims: Fal = False, axis: Optional[OneAx] = None, out: Optional["Ints1d"] = None) -> "Ints1d": ...
    def cumsum(self, *, keepdims: bool = False, axis: _1_AllAx = None, out: Optional["Ints1d"] = None) -> "Ints1d": ...

    @overload
    def sum(self, *, keepdims: Tru, axis: Optional[OneAx] = None, out: Optional["Ints1d"] = None) -> "Ints1d": ...
    @overload
    def sum(self, *, keepdims: Fal = False, axis: Optional[OneAx] = None, out = None) -> int: ...
    def sum(self, *, keepdims: bool = False, axis: _1_AllAx = None, out: Optional["Ints1d"] = None) -> _1I_ReduceResults: ...



class _Array2d(_Array):
    @classmethod
    def __get_validators__(cls):
        """Runtime validation for pydantic."""
        yield lambda v: validate_array(v, ndim=2)

    @property
    def ndim(self) -> Literal[2]: ...
    @property
    def shape(self) -> Tuple[int, int]: ...

    def __iter__(self) -> Iterator[Array1d]: ...
    def astype(self, dtype: DTypes, order: str = ..., casting: str = ..., subok: bool = ..., copy: bool = ...) -> "Array2d": ...
    # These is actually a bit too strict: It's legal to say 'array2d + array3d'
    # That's kind of bad code though; it's better to write array3d + array2d.
    # We could relax this, but let's try the strict version.
    def __add__(self: ArrayT, other: Union[float, int, Array1d, "Array2d"]) -> ArrayT: ...
    def __sub__(self: ArrayT, other: Union[float, int, Array1d, "Array2d"]) -> ArrayT: ...
    def __mul__(self: ArrayT, other: Union[float, int, Array1d, "Array2d"]) -> ArrayT: ...
    def __pow__(self: ArrayT, other: Union[float, int, Array1d, "Array2d"]) -> ArrayT: ...
    def __matmul__(self: ArrayT, other: Union[float, int, Array1d, "Array2d"]) -> ArrayT: ...
    # These are not too strict though: you can't do += with higher dimensional.
    def __iadd__(self, other: Union[float, int, Array1d, "Array2d"]): ...
    def __isub__(self, other: Union[float, int, Array1d, "Array2d"]): ...
    def __imul__(self, other: Union[float, int, Array1d, "Array2d"]): ...
    def __ipow__(self, other: Union[float, int, Array1d, "Array2d"]): ...

    @overload
    def argmax(self, keepdims: Fal = False, axis: int = -1, out: Optional[_Array] = None) -> Ints1d: ...
    @overload
    def argmax(self, keepdims: Tru, axis: int = -1, out: Optional[_Array] = None) -> "Ints2d": ...
    def argmax(self, keepdims: bool = False, axis: int = -1, out: Optional[_Array] = None) -> Union[Ints1d, "Ints2d"]: ...

    @overload
    def mean(self, keepdims: Fal = False, axis: int = -1, dtype: Optional[DTypes] = None, out: Optional["Floats2d"] = None) -> Floats1d: ...
    @overload
    def mean(self, keepdims: Tru, axis: int = -1, dtype: Optional[DTypes] = None, out: Optional["Floats2d"] = None) -> "Floats2d": ...
    def mean(self, keepdims: bool = False, axis: int = -1, dtype: Optional[DTypes] = None, out: Optional["Floats2d"] = None) -> Union["Floats2d", Floats1d]: ...


class Floats2d(_Array2d, _Floats):
    """2-dimensional array of floats"""

    T: "Floats2d"

    @classmethod
    def __get_validators__(cls):
        """Runtime validation for pydantic."""
        yield lambda v: validate_array(v, ndim=2, dtype="f")

    def __iter__(self) -> Iterator[Floats1d]: ...

    @overload
    def __getitem__(self, key: _2_KeyScalar) -> float: ...
    @overload
    def __getitem__(self, key: _2_Key1d) -> Floats1d: ...
    @overload
    def __getitem__(self, key: _2_Key2d) -> "Floats2d": ...
    def __getitem__(self, key: _2_AllKeys) -> _F2_AllReturns: ...

    @overload
    def __setitem__(self, key: _2_KeyScalar, value: float) -> None: ...
    @overload
    def __setitem__(self, key: _2_Key1d, value: Union[float, Floats1d]) -> None: ...
    @overload
    def __setitem__(self, key: _2_Key2d, value: _F2_AllReturns) -> None: ...
    def __setitem__(self, key: _2_AllKeys, value: _F2_AllReturns) -> None: ...

    @overload
    def sum(self, *, keepdims: Tru, axis: _2_AllAx = None, out: Optional["Floats2d"] = None) -> "Floats2d": ...
    @overload
    def sum(self, *, keepdims: Fal = False, axis: OneAx, out: Optional[Floats1d] = None) -> Floats1d: ...
    @overload
    def sum(self, *, keepdims: Fal = False, axis: TwoAx, out = None) -> float: ...
    def sum(self, *, keepdims: bool = False, axis: _2_AllAx = None, out: Union[None, "Floats1d", "Floats2d"] = None) -> _2F_ReduceResults: ...



class Ints2d(_Array2d, _Ints):
    """2-dimensional array of ints."""

    T: "Ints2d"

    @classmethod
    def __get_validators__(cls):
        """Runtime validation for pydantic."""
        yield lambda v: validate_array(v, ndim=2, dtype="i")

    def __iter__(self) -> Iterator[Ints1d]: ...

    @overload
    def __getitem__(self, key: _2_KeyScalar) -> int: ...
    @overload
    def __getitem__(self, key: _2_Key1d) -> Ints1d: ...
    @overload
    def __getitem__(self, key: _2_Key2d) -> "Ints2d": ...
    def __getitem__(self, key: _2_AllKeys) -> _I2_AllReturns: ...

    @overload
    def __setitem__(self, key: _2_KeyScalar, value: int) -> None: ...
    @overload
    def __setitem__(self, key: _2_Key1d, value: Ints1d) -> None: ...
    @overload
    def __setitem__(self, key: _2_Key2d, value: "Ints2d") -> None: ...
    def __setitem__(self, key: _2_AllKeys, value: _I2_AllReturns) -> None: ...

    @overload
    def sum(self, keepdims: Fal = False, axis: int = -1, out: Optional["Ints1d"] = None) -> Ints1d: ...
    @overload
    def sum(self, keepdims: Tru, axis: int = -1, out: Optional["Ints2d"] = None) -> "Ints2d": ...
    def sum(self, keepdims: bool = False, axis: int = -1, out: Optional[Union["Ints1d", "Ints2d"]] = None) -> Union["Ints2d", Ints1d]: ...


class _Array3d(_Array):
    """3-dimensional array of floats"""

    @classmethod
    def __get_validators__(cls):
        """Runtime validation for pydantic."""
        yield lambda v: validate_array(v, ndim=3)

    @property
    def ndim(self) -> Literal[3]: ...
    @property
    def shape(self) -> Tuple[int, int, int]: ...

    def __iter__(self) -> Iterator[Array2d]: ...
    def astype(self, dtype: DTypes, order: str = ..., casting: str = ..., subok: bool = ..., copy: bool = ...) -> "Array3d": ...
    # These is actually a bit too strict: It's legal to say 'array2d + array3d'
    # That's kind of bad code though; it's better to write array3d + array2d.
    # We could relax this, but let's try the strict version.
    def __add__(self: SelfT, other: Union[float, int, Array1d, Array2d, "Array3d"]) -> SelfT: ...
    def __sub__(self: SelfT, other: Union[float, int, Array1d, Array2d, "Array3d"]) -> SelfT: ...
    def __mul__(self: SelfT, other: Union[float, int, Array1d, Array2d, "Array3d"]) -> SelfT: ...
    def __pow__(self: SelfT, other: Union[float, int, Array1d, Array2d, "Array3d"]) -> SelfT: ...
    def __matmul__(self: SelfT, other: Union[float, int, Array1d, Array2d, "Array3d"]) -> SelfT: ...
    # These are not too strict though: you can't do += with higher dimensional.
    def __iadd__(self, other: Union[float, int, Array1d, Array2d, "Array3d"]): ...
    def __isub__(self, other: Union[float, int, Array1d, Array2d, "Array3d"]): ...
    def __imul__(self, other: Union[float, int, Array1d, Array2d, "Array3d"]): ...
    def __ipow__(self, other: Union[float, int, Array1d, Array2d, "Array3d"]): ...

    @overload
    def argmax(self, keepdims: Fal = False, axis: int = -1, out: Optional[_Array] = None) -> Ints2d: ...
    @overload
    def argmax(self, keepdims: Tru, axis: int = -1, out: Optional[_Array] = None) -> "Ints3d": ...
    def argmax(self, keepdims: bool = False, axis: int = -1, out: Optional[_Array] = None) -> Union[Ints2d, "Ints3d"]: ...


class Floats3d(_Array3d, _Floats):
    """3-dimensional array of floats"""

    T: "Floats3d"

    @classmethod
    def __get_validators__(cls):
        """Runtime validation for pydantic."""
        yield lambda v: validate_array(v, ndim=3, dtype="f")

    def __iter__(self) -> Iterator[Floats2d]: ...

    @overload
    def __getitem__(self, key: _3_KeyScalar) -> float: ...
    @overload
    def __getitem__(self, key: _3_Key1d) -> Floats1d: ...
    @overload
    def __getitem__(self, key: _3_Key2d) -> Floats2d: ...
    @overload
    def __getitem__(self, key: _3_Key3d) -> "Floats3d": ...
    def __getitem__(self, key: _3_AllKeys) -> _F3_AllReturns: ...

    @overload
    def __setitem__(self, key: _3_KeyScalar, value: float) -> None: ...
    @overload
    def __setitem__(self, key: _3_Key1d, value: Floats1d) -> None: ...
    @overload
    def __setitem__(self, key: _3_Key2d, value: Floats2d) -> None: ...
    @overload
    def __setitem__(self, key: _3_Key3d, value: "Floats3d") -> None: ...
    def __setitem__(self, key: _3_AllKeys, value: _F3_AllReturns) -> None: ...

    @overload
    def sum(self, *, keepdims: Tru, axis: _3_AllAx = None, out: Optional["Floats3d"] = None) -> "Floats3d": ...
    @overload
    def sum(self, *, keepdims: Fal, axis: OneAx, out: Optional[Floats2d] = None) -> Floats2d: ...
    @overload
    def sum(self, *, keepdims: Fal, axis: TwoAx, out: Optional[Floats1d] = None) -> Floats1d: ...
    @overload
    def sum(self, *, keepdims: Fal, axis: Optional[ThreeAx], out = None) -> float: ...
    def sum(self, *, keepdims: bool = False, axis: _3_AllAx = None, out: Union[None, Floats1d, Floats2d, "Floats3d"] = None) -> _3F_ReduceResults: ...


class Ints3d(_Array3d, _Ints):
    """3-dimensional array of ints."""

    T: "Ints3d"

    @classmethod
    def __get_validators__(cls):
        """Runtime validation for pydantic."""
        yield lambda v: validate_array(v, ndim=3, dtype="i")

    def __iter__(self) -> Iterator[Ints2d]: ...

    @overload
    def __getitem__(self, key: _3_KeyScalar) -> int: ...
    @overload
    def __getitem__(self, key: _3_Key1d) -> Ints1d: ...
    @overload
    def __getitem__(self, key: _3_Key2d) -> Ints2d: ...
    @overload
    def __getitem__(self, key: _3_Key3d) -> "Ints3d": ...
    def __getitem__(self, key: _3_AllKeys) -> _I3_AllReturns: ...

    @overload
    def __setitem__(self, key: _3_KeyScalar, value: int) -> None: ...
    @overload
    def __setitem__(self, key: _3_Key1d, value: Ints1d) -> None: ...
    @overload
    def __setitem__(self, key: _3_Key2d, value: Ints2d) -> None: ...
    @overload
    def __setitem__(self, key: _3_Key3d, value: "Ints3d") -> None: ...
    def __setitem__(self, key: _3_AllKeys, value: _I3_AllReturns) -> None: ...

    @overload
    def sum(self, *, keepdims: Tru, axis: _3_AllAx = None, out: Optional["Ints3d"] = None) -> "Ints3d": ...
    @overload
    def sum(self, *, keepdims: Fal, axis: OneAx, out: Optional[Ints2d] = None) -> Ints2d: ...
    @overload
    def sum(self, *, keepdims: Fal, axis: TwoAx, out: Optional[Ints1d] = None) -> Ints1d: ...
    @overload
    def sum(self, *, keepdims: Fal, axis: Optional[ThreeAx], out = None) -> int: ...
    def sum(self, *, keepdims: bool = False, axis: _3_AllAx = None, out: Union[None, Ints1d, Ints2d, "Ints3d"] = None) -> _3I_ReduceResults: ...


class _Array4d(_Array):
    """4-dimensional array."""

    @classmethod
    def __get_validators__(cls):
        """Runtime validation for pydantic."""
        yield lambda v: validate_array(v, ndim=4)

    @property
    def ndim(self) -> Literal[4]: ...
    @property
    def shape(self) -> Tuple[int, int, int, int]: ...

    def __iter__(self) -> Iterator[Array3d]: ...
    def astype(self, dtype: DTypes, order: str = ..., casting: str = ..., subok: bool = ..., copy: bool = ...) -> "_Array4d": ...
    # These is actually a bit too strict: It's legal to say 'array4d + array5d'
    # That's kind of bad code though; it's better to write array5d + array4d.
    # We could relax this, but let's try the strict version.
    def __add__(self: SelfT, other: Union[float, int, Array1d, Array2d, Array3d, "Array4d"]) -> SelfT: ...
    def __sub__(self: SelfT, other: Union[float, int, Array1d, Array2d, Array3d, "Array4d"]) -> SelfT: ...
    def __mul__(self: SelfT, other: Union[float, int, Array1d, Array2d, Array3d, "Array4d"]) -> SelfT: ...
    def __pow__(self: SelfT, other: Union[float, int, Array1d, Array2d, Array3d, "Array4d"]) -> SelfT: ...
    def __matmul__(self: SelfT, other: Union[float, int, Array1d, Array2d, Array3d, "Array4d"]) -> SelfT: ...
    # These are not too strict though: you can't do += with higher dimensional.
    def __iadd__(self, other: Union[float, int, Array1d, Array2d, Array3d, "Array4d"]): ...
    def __isub__(self, other: Union[float, int, Array1d, Array2d, Array3d, "Array4d"]): ...
    def __imul__(self, other: Union[float, int, Array1d, Array2d, Array3d, "Array4d"]): ...
    def __ipow__(self, other: Union[float, int, Array1d, Array2d, Array3d, "Array4d"]): ...


class Floats4d(_Array4d, _Floats):
    """4-dimensional array of floats."""

    T: "Floats4d"

    @classmethod
    def __get_validators__(cls):
        """Runtime validation for pydantic."""
        yield lambda v: validate_array(v, ndim=4, dtype="f")

    def __iter__(self) -> Iterator[Floats3d]: ...
    # def __getitem__(self, key: int) -> Floats3d: ...

    @overload
    def sum(self, *, keepdims: Tru, axis: _4_AllAx = None, out: Optional["Floats4d"] = None) -> "Floats4d": ...
    @overload
    def sum(self, *, keepdims: Fal = False, axis: OneAx, out: Optional[Floats3d] = None) -> Floats3d: ...
    @overload
    def sum(self, *, keepdims: Fal = False, axis: TwoAx, out: Optional[Floats2d] = None) -> Floats2d: ...
    @overload
    def sum(self, *, keepdims: Fal = False, axis: ThreeAx, out: Optional[Floats1d] = None) -> Floats1d: ...
    @overload
    def sum(self, *, keepdims: Fal = False, axis: Optional[FourAx], out = None) -> float: ...
    def sum(self, *, keepdims: bool = False, axis: _4_AllAx = None, out: Union[None, Floats1d, Floats2d, Floats3d, "Floats4d"] = None) -> _4F_ReduceResults: ...



class Ints4d(_Array4d, _Ints):
    """4-dimensional array of ints."""

    T: "Ints4d"

    @classmethod
    def __get_validators__(cls):
        """Runtime validation for pydantic."""
        yield lambda v: validate_array(v, ndim=4, dtype="i")

    def __iter__(self) -> Iterator[Ints3d]: ...
    # def __getitem__(self, key: int) -> Ints3d: ...

    @overload
    def sum(self, *, keepdims: Tru, axis: _4_AllAx = None, out: Optional["Ints4d"] = None) -> "Ints4d": ...
    @overload
    def sum(self, *, keepdims: Fal = False, axis: OneAx, out: Optional[Ints3d] = None) -> Ints3d: ...
    @overload
    def sum(self, *, keepdims: Fal = False, axis: TwoAx, out: Optional[Ints2d] = None) -> Ints2d: ...
    @overload
    def sum(self, *, keepdims: Fal = False, axis: ThreeAx, out: Optional[Ints1d] = None) -> Ints1d: ...
    @overload
    def sum(self, *, keepdims: Fal = False, axis: Optional[FourAx] = None, out = None) -> int: ...
    def sum(self, *, keepdims: bool = False, axis: _4_AllAx = None, out: Optional[Union[Ints1d, Ints2d, Ints3d, "Ints4d"]] = None) -> _4I_ReduceResults: ...



_DIn = TypeVar("_DIn")


class Decorator(Protocol):
    """Protocol to mark a function as returning its child with identical signature."""

    def __call__(self, name: str) -> Callable[[_DIn], _DIn]: ...


# fmt: on


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

    data: Floats3d
    size_at_t: Ints1d
    lengths: Ints1d
    indices: Ints1d

    def __len__(self) -> int:
        return self.lengths.shape[0]

    def __getitem__(self, index: Union[int, slice, Ints1d]) -> "Padded":
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
    lengths: Ints1d
    data_shape: Tuple[int, ...]
    _cumsums: Optional[Ints1d] = None

    def __init__(self, data: _Array, lengths: Ints1d):
        self.lengths = lengths
        # Frustratingly, the -1 dimension doesn't work with 0 size...
        if data.size:
            self.data = cast(Array2d, data.reshape((data.shape[0], -1)))
        else:
            self.data = cast(Array2d, data.reshape((0, 0)))
        self.data_shape = (-1,) + data.shape[1:]

    @property
    def dataXd(self) -> ArrayXd:
        if self.data.size:
            reshaped = self.data.reshape(self.data_shape)
        else:
            reshaped = self.data.reshape((self.data.shape[0],) + self.data_shape[1:])
        return cast(ArrayXd, reshaped)

    def __len__(self) -> int:
        return self.lengths.shape[0]

    def __getitem__(self, index: Union[int, slice, Array1d]) -> "Ragged":
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
            return Ragged(self.data[start:end].reshape(self.data_shape), lengths)
        else:
            # There must be a way to do this "properly" :(. Sigh, hate numpy.
            xp = get_array_module(self.data)
            data = xp.vstack([self[int(i)].data for i in index])
            return Ragged(data.reshape(self.data_shape), self.lengths[index])

    def _get_cumsums(self) -> Ints1d:
        if self._cumsums is None:
            self._cumsums = self.lengths.cumsum()
        return self._cumsums

    def _get_starts(self) -> Ints1d:
        cumsums = self._get_cumsums()
        xp = get_array_module(cumsums)
        zero = xp.array([0], dtype="i")
        return xp.concatenate((zero, cumsums[:-1]))

    def _get_ends(self) -> Ints1d:
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


@dataclass
class Unserializable:
    """Wrap a value to prevent it from being serialized by msgpack."""

    obj: Any


def validate_array(obj, ndim=None, dtype=None):
    """Runtime validator for pydantic to validate array types."""
    xp = get_array_module(obj)
    if not isinstance(obj, xp.ndarray):
        raise TypeError("not a valid numpy or cupy array")
    errors = []
    if ndim is not None and obj.ndim != ndim:
        errors.append(f"wrong array dimensions (expected {ndim}, got {obj.ndim})")
    if dtype is not None:
        dtype_mapping = {"f": ["float32"], "i": ["int32", "int64", "uint32", "uint64"]}
        expected_types = dtype_mapping.get(dtype, [])
        if obj.dtype not in expected_types:
            expected = "/".join(expected_types)
            err = f"wrong array data type (expected {expected}, got {obj.dtype})"
            errors.append(err)
    if errors:
        raise ValueError(", ".join(errors))
    return obj
