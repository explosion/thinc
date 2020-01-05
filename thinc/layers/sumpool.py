from typing import Callable, TypeVar, Tuple

from ..data import Ragged
from ..model import Model
from ..types import Array


InT = TypeVar("InT", bound=Ragged)
OutT = TypeVar("OutT", bound=Array)


def SumPool() -> Model:
    return Model("sum_pool", forward)


def forward(model: Model, Xr: InT, is_train: bool) -> Tuple[OutT, Callable]:
    Y = model.ops.sum_pool(Xr.data, Xr.lengths)
    lengths = Xr.lengths

    def backprop(dY: OutT) -> InT:
        return Ragged(model.ops.backprop_sum_pool(dY, lengths), lengths)

    return Y, backprop
