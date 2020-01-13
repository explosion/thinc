from typing import Tuple, Callable, Optional, TypeVar

from ..model import Model
from ..config import registry
from ..types import Array
from ..util import get_width
from thinc.types import Reduced_OutT


InT = TypeVar("InT", bound=Array)


@registry.layers("add.v0")
def registry_add(*layer: Model) -> Model:
    return add(*layer)


def add(layer1: Model[InT, InT], layer2: Model[InT, InT], *layers: Model) -> Model[InT, Reduced_OutT]:
    """Compose two or more models `f`, `g`, etc, such that their outputs are
    added, i.e. `add(f, g)(x)` computes `f(x) + g(x)`.
    """
    layers = (layer1, layer2) + layers
    if layers[0].name == "add":
        layers[0].layers.extend(layers[1:])
        return layers[0]
    return Model(
        "add", forward, init=init, dims={"nO": None, "nI": None}, layers=layers
    )


def forward(model: Model[InT, InT], X: InT, is_train: bool) -> Tuple[InT, Callable]:
    if not model.layers:
        return X, lambda dY: dY
    Y, first_callback = model.layers[0](X, is_train=is_train)
    callbacks = []
    for layer in model.layers[1:]:
        layer_Y, layer_callback = layer(X, is_train=is_train)
        Y += layer_Y
        callbacks.append(layer_callback)

    def backprop(dY: InT) -> InT:
        dX = first_callback(dY)
        for callback in callbacks:
            dX += callback(dY)
        return dX

    return Y, backprop


def init(
    model: Model[InT, InT], X: Optional[InT] = None, Y: Optional[InT] = None
) -> None:
    if X is not None:
        X_width = get_width(X)
        model.set_dim("nI", X_width)
        for layer in model.layers:
            layer.set_dim("nI", X_width)
    for layer in model.layers:
        layer.initialize(X=X, Y=Y)
    model.set_dim("nO", model.layers[0].get_dim("nO"))
