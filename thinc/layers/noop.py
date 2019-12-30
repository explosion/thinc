from typing import Tuple, Callable, TypeVar

from ..model import Model


InputType = TypeVar("InputType")
OutputType = TypeVar("OutputType")


def noop(*layers: Model) -> Model:
    """Transform a sequences of layers into a null operation."""
    return Model("noop", forward, layers=layers)


def forward(model: Model, X: InputType, is_train: bool) -> Tuple[OutputType, Callable]:
    def backprop(dY: OutputType) -> InputType:
        return dY

    return X, backprop
