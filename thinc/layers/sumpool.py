from typing import Callable, Tuple

from ..model import Model
from ..config import registry
from ..types import Array, Ragged


InT = Ragged
OutT = Array


@registry.layers("SumPool.v0")
def SumPool() -> Model[InT, OutT]:
    return Model("sum_pool", forward)


def forward(model: Model[InT, OutT], Xr: InT, is_train: bool) -> Tuple[OutT, Callable]:
    Y = model.ops.sum_pool(Xr.data, Xr.lengths)
    lengths = Xr.lengths

    def backprop(dY: OutT) -> InT:
        return Ragged(model.ops.backprop_sum_pool(dY, lengths), lengths)

    return Y, backprop
