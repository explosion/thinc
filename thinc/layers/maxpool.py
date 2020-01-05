from typing import Tuple, Callable, TypeVar

from ..types import Array
from ..model import Model
from ..data import Ragged


InT = TypeVar("InT", bound=Ragged)
OutT = TypeVar("OutT", bound=Array)


def MaxPool() -> Model:
    return Model("max_pool", forward)


def forward(model: Model, Xr: InT, is_train: bool) -> Tuple[OutT, Callable]:
    Y, which = model.ops.max_pool(Xr.data, Xr.lengths)
    lengths = Xr.lengths

    def backprop(dY: OutT) -> InT:
        return Ragged(model.ops.backprop_max_pool(dY, which, lengths), lengths)

    return Y, backprop
