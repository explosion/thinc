from typing import Tuple, Callable, TypeVar

from ..types import Array
from ..data import Ragged
from ..model import Model


InT = TypeVar("InT", bound=Ragged)
OutT = TypeVar("OutT", bound=Array)


def MeanPool() -> Model:
    return Model("mean_pool", forward)


def forward(model: Model, Xr: InT, is_train: bool) -> Tuple[OutT, Callable]:
    Y = model.ops.mean_pool(Xr.data, Xr.lengths)
    lengths = Xr.lengths

    def backprop(dY: OutT) -> InT:
        return Ragged(model.ops.backprop_mean_pool(dY, lengths), lengths)

    return Y, backprop
