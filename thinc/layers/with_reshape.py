from typing import Tuple, Callable, Optional

from .base import Model, Array


def with_reshape(layer: Model) -> Model:
    """Reshape data on the way into and out from a layer."""
    return Model(
        f"with_reshape-{layer.name}",
        forward,
        init=init,
        layers=[layer],
        dims={"nO": None, "nI": None},
    )


def forward(model: Model, X: Array, is_train: bool) -> Tuple[Array, Callable]:
    layer = model.layers[0]
    initial_shape = X.shape
    final_shape = list(initial_shape[:-1]) + [layer.get_dim("nO")]
    nB = X.shape[0]
    nT = X.shape[1]
    X2d = X.reshape(-1, X.shape[2])
    X2d = X2d.astype(layer.ops.xp.float32)
    Y2d, Y2d_backprop = layer(X2d, is_train=is_train)
    Y = Y2d.reshape(final_shape)

    def with_reshape_backward(dY):
        dY = dY.reshape(nB * nT, -1).astype(layer.ops.xp.float32)
        return Y2d_backprop(dY).reshape(initial_shape)

    return Y, with_reshape_backward


def init(model: Model, X: Optional[Array] = None, Y: Optional[Array] = None) -> None:
    # TODO: write
    pass
