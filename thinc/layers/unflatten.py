from typing import Tuple, Callable, TypeVar, List

from ..model import Model
from ..types import Array


InputValue = TypeVar("InputValue", bound=Array)
InputLengths = TypeVar("InputLengths", bound=Array)
InputType = Tuple[InputValue, InputLengths]
OutputValue = TypeVar("OutputValue", bound=Array)
OutputType = List[OutputValue]


def unflatten() -> Model:
    """Transform sequences from a ragged format into lists."""
    return Model("unflatten", forward)


def forward(
    model: Model, X_lengths: InputType, is_train: bool
) -> Tuple[OutputType, Callable]:
    X, lengths = X_lengths
    Xs = model.ops.unflatten(X, lengths)

    def backprop(dXs: OutputType) -> InputType:
        return model.ops.flatten(dXs, pad=0), lengths

    return Xs, backprop
