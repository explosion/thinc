from typing import Callable, Tuple, cast

from ..model import Model
from ..config import registry
from ..types import Floats2d, Ragged


@registry.layers("reduce_last.v1")
def reduce_last() -> Model[Ragged, Floats2d]:
    return Model("reduce_last", forward)


def forward(model: Model[Ragged, Floats2d], Xr: Ragged, is_train: bool) -> Tuple[Floats2d, Callable]:
    ends = Xr.lengths - 1
    ends[1:] += Xr.lengths.cumsum()[:-1]
    Y = Xr.dataXd[ends]
    x_shape = Xr.dataXd.shape
    lengths = Xr.lengths

    def backprop(dY: Floats2d) -> Ragged:
        dX = model.ops.alloc2f(*x_shape, dtype=dY.dtype)
        dX[ends] = dY
        return Ragged(dX, lengths)

    return Y, backprop
