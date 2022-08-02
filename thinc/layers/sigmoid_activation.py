from typing import TypeVar, Tuple, Callable, cast

from ..model import Model
from ..config import registry
from ..types import FloatsXdT


@registry.layers("sigmoid_activation.v1")
def sigmoid_activation() -> Model[FloatsXdT, FloatsXdT]:
    return Model("sigmoid_activation", forward)


def forward(
    model: Model[FloatsXdT, FloatsXdT], X: FloatsXdT, is_train: bool
) -> Tuple[FloatsXdT, Callable]:
    Y = model.ops.sigmoid(X, inplace=False)

    def backprop(dY: FloatsXdT) -> FloatsXdT:
        return cast(
            FloatsXdT,
            dY * model.ops.dsigmoid(Y, inplace=False),  # type:ignore[operator]
        )

    return Y, backprop
