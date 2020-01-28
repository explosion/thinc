from typing import Union, Sequence, Tuple
from ..types import ArrayXd, FloatsXd, IntsXd
from ..model import Model


AxisIndex = Union[int, slice, Sequence[int]]
Index = Union[AxisIndex, Tuple[AxisIndex, ...]]


def array_getitem(index: Index) -> Model[ArrayXd, ArrayXd]:
    """Index into input arrays, and return the subarrays.

    The `index` object can
    """
    return Model("array-getitem", forward, attrs={"index": index})


def floats_getitem(index: Index) -> Model[FloatsXd, FloatsXd]:
    """Index into input arrays, and return the subarrays.

    This delegates to `array_getitem`, but allows type declarations.
    """
    return Model("floats-getitem", forward, attrs={"index": index})


def ints_getitem(index: Index) -> Model[IntsXd, IntsXd]:
    """Index into input arrays, and return the subarrays.

    This delegates to `array_getitem`, but allows type declarations.
    """
    return Model("ints-getitem", forward, attrs={"index": index})


def forward(model, X, is_train):
    index = model.attrs["index"]
    shape = X.shape
    dtype = X.dtype

    def backprop_get_column(dY):
        dX = model.ops.alloc(shape, dtype=dtype)
        dX[index] = dY
        return dX

    Y = X[index]
    return Y, backprop_get_column
