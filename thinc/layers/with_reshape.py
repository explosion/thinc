from typing import Tuple, Callable, Optional, cast

from ..model import Model
from ..types import Floats3d, Floats2d


InT = Floats3d


def with_reshape(layer: Model[Floats2d, Floats2d]) -> Model[InT, InT]:
    """Reshape data on the way into and out from a layer."""
    return Model(
        f"with_reshape-{layer.name}",
        forward,
        init=init,
        layers=[layer],
        dims={"nO": None, "nI": None},
    )


def forward(model: Model[InT, InT], X: InT, is_train: bool) -> Tuple[InT, Callable]:
    layer = model.layers[0]
    initial_shape = X.shape
    final_shape = list(initial_shape[:-1]) + [layer.get_dim("nO")]
    nB = X.shape[0]
    nT = X.shape[1]
    X2d = X.reshape((-1, X.shape[2]))
    X2d = X2d.astype(layer.ops.xp.float32)
    Y2d, Y2d_backprop = layer(X2d, is_train=is_train)
    Y = Y2d.reshape(final_shape)

    def backprop(dY: InT) -> InT:
        dY = cast(Floats3d, dY.reshape((nB * nT, -1)).astype(layer.ops.xp.float32))
        return Y2d_backprop(dY).reshape(initial_shape)

    return Y, backprop


def init(
    model: Model[InT, InT], X: Optional[Floats3d] = None, Y: Optional[Floats3d] = None
) -> None:
    layer = model.layers[0]
    if X is None and Y is None:
        layer.initialize()
        return
    X2d: Optional[Floats2d] = None
    Y2d: Optional[Floats2d] = None
    if X is not None:
        X2d = cast(Floats2d, X.reshape((-1, X.shape[-1])))
    if Y is not None:
        Y2d = cast(Floats2d, Y.reshape((-1, Y.shape[-1])))
    layer.initialize(X=X2d, Y=Y2d)
    model.set_dim("nI", layer.get_dim("nI"))
    model.set_dim("nO", layer.get_dim("nO"))
