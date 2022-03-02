from typing import Callable, Tuple, cast, TypeVar

from ..model import Model
from ..config import registry
from ..types import Ragged, ArrayXd

OutT = TypeVar("OutT", bound=ArrayXd)

@registry.layers("reduce_first.v1")
def reduce_first() -> Model[Ragged, OutT]:
    """Reduce ragged-formatted sequences to their first element."""
    return Model("reduce_first", forward)


def forward(model: Model[Ragged, OutT], Xr: Ragged, is_train: bool) -> Tuple[OutT, Callable]:
    starts = model.ops.alloc1i(Xr.lengths.shape[0])
    starts[1:] += Xr.lengths.cumsum()[:-1]
    X = cast(OutT, Xr.dataXd)
    Y = cast(OutT, X[starts]) # type: ignore
    x_shape = Xr.dataXd.shape
    lengths = Xr.lengths

    def backprop(dY: OutT) -> Ragged:
        dX = cast(OutT, model.ops.alloc(x_shape, dtype=dY.dtype))
        dX[starts] = dY # type: ignore
        return Ragged(dX, lengths)

    return Y, backprop
