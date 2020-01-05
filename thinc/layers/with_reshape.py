from typing import Tuple, Callable, Optional

from ..model import Model
from ..types import Array


# TODO: more specific types?
InT = Array
OutT = Array


def with_reshape(layer: Model) -> Model[InT, OutT]:
    """Reshape data on the way into and out from a layer."""
    return Model(
        f"with_reshape-{layer.name}",
        forward,
        init=init,
        layers=[layer],
        dims={"nO": None, "nI": None},
    )


def forward(model: Model[InT, OutT], X: InT, is_train: bool) -> Tuple[OutT, Callable]:
    layer = model.layers[0]
    initial_shape = X.shape
    final_shape = list(initial_shape[:-1]) + [layer.get_dim("nO")]
    nB = X.shape[0]
    nT = X.shape[1]
    X2d = X.reshape((-1, X.shape[2]))
    X2d = X2d.astype(layer.ops.xp.float32)
    Y2d, Y2d_backprop = layer(X2d, is_train=is_train)
    Y = Y2d.reshape(final_shape)

    def backprop(dY: OutT) -> InT:
        dY = dY.reshape((nB * nT, -1)).astype(layer.ops.xp.float32)
        return Y2d_backprop(dY).reshape(initial_shape)

    return Y, backprop


def init(
    model: Model[InT, OutT], X: Optional[InT] = None, Y: Optional[OutT] = None
) -> None:
    # TODO: write
    pass
