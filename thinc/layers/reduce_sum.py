from typing import Callable, Tuple

from ..model import Model
from ..config import registry
from ..types import Array2d, Ragged


InT = Ragged
OutT = Array2d


@registry.layers("reduce_sum.v1")
def reduce_sum() -> Model[InT, OutT]:
    return Model("reduce_sum", forward)


def forward(model: Model[InT, OutT], Xr: InT, is_train: bool) -> Tuple[OutT, Callable]:
    Y = model.ops.reduce_sum(Xr.data, Xr.lengths)
    lengths = Xr.lengths

    def backprop(dY: OutT) -> InT:
        return Ragged(model.ops.backprop_reduce_sum(dY, lengths), lengths)

    return Y, backprop
