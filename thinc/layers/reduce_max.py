from typing import Tuple, Callable

from ..types import Array2d, Ragged
from ..model import Model
from ..config import registry


InT = Ragged
OutT = Array2d


@registry.layers("reduce_max.v0")
def reduce_max() -> Model[InT, OutT]:
    return Model("reduce_max", forward)


def forward(model: Model[InT, OutT], Xr: InT, is_train: bool) -> Tuple[OutT, Callable]:
    Y: Array2d
    Y, which = model.ops.reduce_max(Xr.data, Xr.lengths)
    lengths = Xr.lengths

    def backprop(dY: OutT) -> InT:
        return Ragged(model.ops.backprop_reduce_max(dY, which, lengths), lengths)

    return Y, backprop
