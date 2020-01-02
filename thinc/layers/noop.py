from typing import Tuple, Callable, TypeVar

from ..model import Model


InputOutputType = TypeVar("InputOutputType")


def noop(*layers: Model) -> Model:
    """Transform a sequences of layers into a null operation."""
    return Model("noop", forward, layers=layers)


def forward(
    model: Model, X: InputOutputType, is_train: bool
) -> Tuple[InputOutputType, Callable]:
    def backprop(dY: InputOutputType) -> InputOutputType:
        return dY

    return X, backprop
