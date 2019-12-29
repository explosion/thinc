from typing import Tuple, Callable

from .base import Model, Array


def Dropout(rate: float = 0.0) -> Model:
    return Model(
        "dropout",
        forward,
        init=None,
        dims={},
        params={},
        layers=[],
        attrs={"rate": rate, "is_enabled": True},
    )


def forward(model: Model, X: Array, is_train: bool) -> Tuple[Array, Callable]:
    rate = model.get_attr("rate")
    is_enabled = model.get_attr("is_enabled")
    if not is_enabled:
        return X, lambda dY: dY
    elif isinstance(X, tuple) and len(X) == 2:
        Y, wrap_backprop = model.ops.dropout(X[0], rate, inplace=False)
        return (Y, X[1]), wrap_backprop(lambda dY: dY)
    elif isinstance(X, list):
        Y, wrap_backprop = model.ops.dropout_sequences(X, rate, inplace=False)
        return Y, wrap_backprop(lambda dY: dY)
    else:
        Y, wrap_backprop = model.ops.dropout(X, rate, inplace=False)
        return Y, wrap_backprop(lambda dY: dY)
