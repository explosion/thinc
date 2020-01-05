from typing import Tuple, Callable, TypeVar

from ..model import Model


InputOutT = TypeVar("InputOutT")


def noop(*layers: Model) -> Model:
    """Transform a sequences of layers into a null operation."""
    return Model("noop", forward, layers=layers)


def forward(
    model: Model, X: InputOutT, is_train: bool
) -> Tuple[InputOutT, Callable]:
    def backprop(dY: InputOutT) -> InputOutT:
        return dY

    return X, backprop
