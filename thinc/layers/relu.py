from typing import Tuple, Callable, Optional

from .base import Model, Array
from ..util import get_width


def ReLu() -> Model:
    return Model(
        "relu",
        forward,
        init=init,
        dims={"nO": None, "nI": None},
        params={},
        attrs={},
        layers=[],
    )


def forward(model: Model, X: Array, is_train: bool) -> Tuple[Array, Callable]:
    Y = model.ops.relu(X)

    def relu_backward(dY: Array) -> Array:
        return model.ops.backprop_relu(dY, Y)

    return Y, relu_backward


def init(model: Model, X: Optional[Array] = None, Y: Optional[Array] = None) -> None:
    if X is not None:
        X_width = get_width(X)
        model.set_dim("nI", X_width)
        model.set_dim("nO", X_width)
    elif Y is not None:
        Y_width = get_width(Y)
        model.set_dim("nI", Y_width)
        model.set_dim("nO", Y_width)
