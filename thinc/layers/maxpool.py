from typing import Tuple, Callable, TypeVar

from ..types import Array
from ..model import Model
from ..data import Ragged


InputType = TypeVar("InputType", bound=Ragged)
OutputType = TypeVar("OutputType", bound=Array)


def MaxPool() -> Model:
    return Model("max_pool", forward)


def forward(model: Model, Xr: InputType, is_train: bool) -> Tuple[OutputType, Callable]:
    Y, which = model.ops.max_pool(Xr.data, Xr.lengths)
    lengths = Xr.lengths

    def backprop(dY: OutputType) -> InputType:
        return Ragged(model.ops.backprop_max_pool(dY, which, lengths), lengths)

    return Y, backprop
