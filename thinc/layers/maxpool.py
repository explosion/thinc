from typing import Tuple, Callable, TypeVar

from ..types import Array
from ..model import Model


InputValue = TypeVar("InputValue", bound=Array)
InputLengths = TypeVar("InputLengths", bound=Array)
InputType = Tuple[InputValue, InputLengths]
OutputValue = TypeVar("OutputValue", bound=Array)
OutputLengths = TypeVar("OutputLengths", bound=Array)
OutputType = Tuple[OutputValue, OutputLengths]


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
