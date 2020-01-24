from typing import Tuple, Callable

from ..types import Array2d, Ragged
from ..model import Model
from ..config import registry


InT = Ragged
OutT = Array2d


@registry.layers("reduce_mean.v0")
def reduce_mean() -> Model[InT, OutT]:
    return Model("reduce_mean", forward)


def forward(model: Model[InT, OutT], Xr: InT, is_train: bool) -> Tuple[OutT, Callable]:
    Y = model.ops.reduce_mean(Xr.data, Xr.lengths)
    lengths = Xr.lengths

    def backprop(dY: OutT) -> InT:
        return Ragged(model.ops.backprop_reduce_mean(dY, lengths), lengths)

    return Y, backprop
