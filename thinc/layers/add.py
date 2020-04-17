from typing import Tuple, Callable, Optional, TypeVar, Dict

from ..model import Model
from ..config import registry
from ..types import ArrayXd, XY_XY_OutT
from ..util import get_width


InT = TypeVar("InT", bound=ArrayXd)
OutT = TypeVar("OutT", bound=ArrayXd)


@registry.layers("add.v1")
def add(
    layer1: Model[InT, OutT], layer2: Model[InT, OutT], *layers: Model
) -> Model[InT, XY_XY_OutT]:
    """Compose two or more models `f`, `g`, etc, such that their outputs are
    added, i.e. `add(f, g)(x)` computes `f(x) + g(x)`.
    """
    layers = (layer1, layer2) + layers
    if layers[0].name == "add":
        layers[0].layers.extend(layers[1:])
        return layers[0]

    # only add an nI dimension if each sub-layer has one
    dims: Dict[str, Optional[int]] = {"nO": None}
    if all(node.has_dim("nI") in [True, None] for node in layers):
        dims = {"nO": None, "nI": None}

    return Model("add", forward, init=init, dims=dims, layers=layers)


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
) -> Model[InT, InT]:
    if X is not None:
        if model.has_dim("nI") is not False:
            model.set_dim("nI", get_width(X))
        for layer in model.layers:
            if layer.has_dim("nI") is not False:
                layer.set_dim("nI", get_width(X))
    for layer in model.layers:
        layer.initialize(X=X, Y=Y)
    model.set_dim("nO", model.layers[0].get_dim("nO"))
    return model
