from typing import Callable, Optional, Tuple, Any, TypeVar

from ..model import Model
from ..config import registry

InT = TypeVar("InT")
OutT = TypeVar("OutT")
MidT = TypeVar("MidT")


@registry.layers("tuplify.v1")
def tuplify(layer1: Model, layer2: Model, *layers) -> Model:
    layers = (layer1, layer2) + layers
    names = [layer.name for layer in layers]
    return Model(
            "tuple(" + ", ".join(names) + ")", 
            tuplify_forward, 
            layers=layers,
    )


def tuplify_forward(model, X, is_train):
    Ys = []
    backprops = []
    for layer in model.layers:
        Y, backprop = layer(X, is_train)
        Ys.append(Y)
        backprops.append(backprop)

    def backprop_tuplify(dYs):
        dXs = [bp(dY) for bp, dY in zip(backprops, dYs)]
        dX = dXs[0]
        for dx in dXs[1:]:
            dX += dx
        return dX

    return tuple(Ys), backprop_tuplify
