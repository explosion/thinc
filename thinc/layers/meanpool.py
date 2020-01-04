from typing import Tuple, Callable, TypeVar

from ..types import Array
from ..data import Ragged
from ..model import Model


InputType = TypeVar("InputType", bound=Ragged)
OutputType = TypeVar("OutputType", bound=Array)


def MeanPool() -> Model:
    return Model("mean_pool", forward)


def forward(
    model: Model, Xr: InputType, is_train: bool
) -> Tuple[OutputType, Callable]:
    Y = model.ops.mean_pool(Xr.data, Xr.lengths)
    lengths = Xr.lengths

    def backprop(dY: OutputType) -> InputType:
        return Ragged(model.ops.backprop_mean_pool(dY, lengths), lengths)

    return Y, backprop
