from typing import Tuple, Callable, List, Optional, TypeVar

from ..model import Model
from ..types import Array
from ..util import get_width


InputType = TypeVar("InputType", bound=Array)
OutputType = TypeVar("OutputType", bound=Array)


def add(layers: List[Model]) -> Model:
    """Compose two or more models `f`, `g`, etc, such that their outputs are
    added, i.e. `add(f, g)(x)` computes `f(x) + g(x)`.
    """
    if layers and layers[0].name == "add":
        layers[0].layers.extend(layers[1:])
        return layers[0]
    return Model("add", forward, init=init, dims={"nO": None, "nI": None})


def forward(model: Model, X: InputType, is_train: bool) -> Tuple[OutputType, Callable]:
    if not model.layers:
        return X, lambda dY: dY
    Y, first_callback = model.layers[0](X, is_train=is_train)
    callbacks = []
    for layer in model.layers[1:]:
        layer_Y, layer_callback = layer(X, is_train=is_train)
        Y += layer_Y
        callbacks.append(layer_callback)

    def backprop(dY: OutputType) -> InputType:
        dX = first_callback(dY)
        for callback in callbacks:
            dX += callback(dY)
        return dX

    return Y, backprop


def init(model: Model, X: Optional[Array] = None, Y: Optional[Array] = None) -> None:
    if X is not None:
        X_width = get_width(X)
        model.set_dim("nI", X_width)
        for layer in model.layers:
            layer.set_dim("nI", X_width)
    for layer in model.layers:
        layer.initialize(X=X)
    model.set_dim("nO", sum(layer.get_dim("nO") for layer in model._layers))
