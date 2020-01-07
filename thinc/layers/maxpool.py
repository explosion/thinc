from typing import Tuple, Callable

from ..types import Floats2d, Ragged
from ..model import Model
from ..config import registry


InT = Ragged
OutT = Floats2d


@registry.layers("MaxPool.v0")
def MaxPool() -> Model[InT, OutT]:
    return Model("max_pool", forward)


def forward(model: Model[InT, OutT], Xr: InT, is_train: bool) -> Tuple[OutT, Callable]:
    Y: Floats2d
    Y, which = model.ops.max_pool(Xr.data, Xr.lengths)
    lengths = Xr.lengths

    def backprop(dY: OutT) -> InT:
        return Ragged(model.ops.backprop_max_pool(dY, which, lengths), lengths)

    return Y, backprop
