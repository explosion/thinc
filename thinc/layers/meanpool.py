from typing import Tuple, Callable

from ..types import Floats2d, Ragged
from ..model import Model


InT = Ragged
OutT = Floats2d


def MeanPool() -> Model[InT, OutT]:
    return Model("mean_pool", forward)


def forward(model: Model[InT, OutT], Xr: InT, is_train: bool) -> Tuple[OutT, Callable]:
    Y = model.ops.mean_pool(Xr.data, Xr.lengths)
    lengths = Xr.lengths

    def backprop(dY: OutT) -> InT:
        return Ragged(model.ops.backprop_mean_pool(dY, lengths), lengths)

    return Y, backprop
