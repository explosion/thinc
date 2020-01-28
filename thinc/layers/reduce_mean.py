from typing import Tuple, Callable, cast

from ..types import Floats2d, Ragged
from ..model import Model
from ..config import registry


InT = Ragged
OutT = Floats2d


@registry.layers("reduce_mean.v1")
def reduce_mean() -> Model[InT, OutT]:
    return Model("reduce_mean", forward)


def forward(model: Model[InT, OutT], Xr: InT, is_train: bool) -> Tuple[OutT, Callable]:
    Y = model.ops.reduce_mean(cast(Floats2d, Xr.data), Xr.lengths)
    lengths = Xr.lengths

    def backprop(dY: OutT) -> InT:
        return Ragged(model.ops.backprop_reduce_mean(dY, lengths), lengths)

    return Y, backprop
