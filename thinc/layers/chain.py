from typing import Tuple, Callable, Optional, TypeVar

from ..model import Model
from ..types import Array
from ..util import get_width


InputType = TypeVar("InputType", bound=Array)
OutputType = TypeVar("OutputType", bound=Array)


def chain(*layers: Model) -> Model:
    if layers and layers[0]._func is forward:
        layers[0].layers.extend(layers[1:])
        return layers[0]

    model = Model(
        ">>".join(layer.name for layer in layers),
        forward,
        init=init,
        dims={"nO": None, "nI": None},
        layers=layers,
    )
    if layers and layers[0].get_dim("nI") and layers[-1].get_dim("nO"):
        model.initialize()
    return model


def forward(model: Model, X: InputType, is_train: bool) -> Tuple[OutputType, Callable]:
    """Apply the layers of `model` in sequence, feeding the output from one
    layer into the next.

    Returns (tuple):
        The output of the model, and a callback to complete the backward pass.
    """
    callbacks = []
    for layer in model.layers:
        X, inc_layer_grad = layer(X, is_train=is_train)
        callbacks.append(inc_layer_grad)

    def backprop(gradient: OutputType) -> InputType:
        for callback in reversed(callbacks):
            gradient = callback(gradient)
        return gradient

    return X, backprop


def init(
    model: Model, X: Optional[InputType] = None, Y: Optional[OutputType] = None
) -> None:
    if not model.layers:
        return
    # Try to set nO on each layer, where available.
    nO = get_width(Y) if Y is not None else model.get_dim("nO")
    for layer in reversed(model.layers):
        if nO is not None and layer.dim_is_unset("nO"):
            layer.set_dim("nO", nO)
        nO = layer.get_dim("nI")
    for layer in model.layers[:-1]:
        layer.initialize(X=X)
        X = layer.predict(X)
    model.layers[-1].initialize(X=X, Y=Y)
    model.set_dim("nI", model.layers[0].get_dim("nI"))
    model.set_dim("nO", model.layers[-1].get_dim("nO"))
