from typing import Callable, TypeVar

from ..data import Ragged
from ..model import Model
from ..types import Array


InputType = TypeVar("InputType", bound=Ragged)
OutputType = TypeVar("OutputType", bound=Array)


def SumPool() -> Model:
    return Model("sum_pool", forward)


def forward(
    model: Model, Xr: InputType, is_train: bool
) -> Tuple[OutputType, Callable]:
    Y = model.ops.sum_pool(Xr.data, Xr.lengths)

    def backprop(dY: OutputType) -> InputType:
        return Ragged(model.ops.backprop_sum_pool(dY, lengths), lengths)

    return Y, backprop
