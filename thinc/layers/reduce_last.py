from typing import Callable, Tuple, cast, TypeVar

from ..model import Model
from ..config import registry
from ..types import Ragged, ArrayXd

OutT = TypeVar("OutT", bound=ArrayXd)

@registry.layers("reduce_last.v1")
def reduce_last() -> Model[Ragged, OutT]:
    """Reduce ragged-formatted sequences to their last element."""
    return Model("reduce_last", forward)


def forward(model: Model[Ragged, OutT], Xr: Ragged, is_train: bool) -> Tuple[OutT, Callable]:
    ends = Xr.lengths.cumsum() - 1
    Y = cast(OutT, Xr.dataXd[ends]) # type: ignore
    x_shape = Xr.dataXd.shape
    lengths = Xr.lengths

    def backprop(dY: OutT) -> Ragged:
        dX = cast(OutT, model.ops.alloc(x_shape, dtype=dY.dtype))
        dX[ends] = dY # type: ignore
        return Ragged(dX, lengths)

    return Y, backprop
