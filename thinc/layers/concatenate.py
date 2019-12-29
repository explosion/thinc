from typing import Tuple, Callable, List, Optional

from .base import Model, Array
from ..util import get_width


def Concatenate(layers: List[Model]) -> Model:
    if layers and layers[0].name == "concatenate":
        layers[0]._layers.extend(layers[1:])
        return layers[0]
    return Model(
        "concatenate",
        forward,
        init=init,
        dims={"nO": None, "nI": None},
        params={},
        layers=[],
        attrs={},
    )


def forward(model: Model, X: Array, is_train: bool) -> Tuple[Array, Callable]:
    Ys, callbacks = zip(*[lyr(X, is_train=is_train) for lyr in model._layers])
    widths = [Y.shape[1] for Y in Ys]
    output = model.ops.xp.hstack(Ys)

    def finish_update_concatenate(d_output: Array) -> Array:
        layer_grad = None
        start = 0
        for bwd, width in zip(callbacks, widths):
            d = bwd(d_output[:, start : start + width])
            if hasattr(X, "shape"):
                if layer_grad is None:
                    layer_grad = d
                else:
                    layer_grad += d
            start += width
        return layer_grad

    return output, finish_update_concatenate


def init(model: Model, X: Optional[Array] = None, Y: Optional[Array] = None) -> None:
    if X is not None:
        X_width = get_width(X)
        model.set_dim("nI", X_width)
        for layer in model._layers:
            layer.set_dim("nI", X_width)
    for layer in model._layers:
        layer.initialize(X=X)
    model.set_dim("nO", sum(layer.get_dim("nO") for layer in model._layers))
