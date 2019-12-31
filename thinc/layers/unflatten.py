from typing import Tuple, Callable, TypeVar

from ..model import Model
from ..types import Array


InputType = TypeVar("InputType", bound=Tuple[Array, Array])
OutputType = TypeVar("OutputType", bound=Tuple[Array, Array])


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
