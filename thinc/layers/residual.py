from typing import Tuple, Callable, Optional

from .base import Model, Array


def forward(model: Model, X: Array, is_train: bool) -> Tuple[Array, Callable]:
    y, bp_y = model._layers[0].begin_update(X)
    if isinstance(X, list):
        output = [X[i] + y[i] for i in range(len(X))]
    elif isinstance(X, tuple) and isinstance(y, tuple) and len(X) == 2:
        # Handle case where we have (data, lengths) tuple
        output = (X[0] + y[0], y[1])
    else:
        output = X + y

    def residual_bwd(d_output: Array) -> Array:
        dX = bp_y(d_output)
        if isinstance(d_output, list) or isinstance(d_output, tuple):
            return [d_output[i] + dX[i] for i in range(len(d_output))]
        else:
            return d_output + dX

    return output, residual_bwd


def init(model: Model, X: Optional[Array] = None, Y: Optional[Array] = None) -> None:
    model._layers[0].initialize(X=X, Y=Y)
    model.set_dim("nO", model._layers[0].get_dim("nO"))
    model.set_dim("nI", model._layers[0].get_dim("nI"))


def Residual(layer: Model) -> Model:
    return Model(
        forward,
        init=init,
        layers=[layer],
        params={},
        dims={"nO": layer.get_dim("nO"), "nI": layer.get_dim("nI")},
        attrs={},
    )
