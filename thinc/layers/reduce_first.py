from typing import Callable, Tuple, cast

from ..model import Model
from ..config import registry
from ..types import Floats2d, Ragged


@registry.layers("reduce_fist.v1")
def reduce_first() -> Model[Ragged, Floats2d]:
    return Model("reduce_first", forward)


def forward(model: Model[Ragged, Floats2d], Xr: Ragged, is_train: bool) -> Tuple[Floats2d, Callable]:
    starts = model.ops.alloc1i(Xr.lengths.shape[0])
    starts[1:] += Xr.lengths.cumsum()[:-1]
    Y = Xr.dataXd[starts]
    x_shape = Xr.dataXd.shape
    lengths = Xr.lengths

    def backprop(dY: Floats2d) -> Ragged:
        dX = model.ops.alloc2f(*x_shape, dtype=dY.dtype)
        dX[starts] = dY
        return Ragged(dX, lengths)

    return Y, backprop
