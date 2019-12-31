from typing import Tuple, Callable, TypeVar

from ..types import Array
from ..model import Model


InputType = TypeVar("InputType", bound=Tuple[Array, Array])
OutputType = TypeVar("OutputType", bound=Tuple[Array, Array])


def MaxPool() -> Model:
    return Model("max_pool", forward)


def forward(
    model: Model, X_lengths: InputType, is_train: bool
) -> Tuple[OutputType, Callable]:
    X, lengths = X_lengths
    Y, which = model.ops.max_pool(X, lengths)

    def backprop(dY: OutputType) -> InputType:
        return model.ops.backprop_max_pool(dY, which, lengths)

    return Y, backprop
