from typing import Callable, Optional, TypeVar

from ..model import Model
from ..config import registry


InT = TypeVar("InT")
OutT = TypeVar("OutT")


@registry.layers("resizable.v1")
def resizable(layer_creation: Callable) -> Model[InT, OutT]:
    """Container that holds one layer that can change dimensions.
    Currently supports layers with `W` and `b` parameters.
    """
    layer = layer_creation()
    return Model(
        f"resizable({layer.name})",
        forward,
        init=init,
        layers=[layer],
        attrs={"layer_creation": layer_creation, "resize_output": resize},
        dims={name: layer.maybe_get_dim(name) for name in layer.dim_names},
    )


def forward(model: Model[InT, OutT], X: InT, is_train: bool):
    layer = model.layers[0]
    Y, callback = layer(X, is_train=is_train)

    def backprop(dY: OutT) -> InT:
        return callback(dY)

    return Y, backprop


def init(
    model: Model[InT, OutT], X: Optional[InT] = None, Y: Optional[OutT] = None
) -> Model[InT, OutT]:
    layer = model.layers[0]
    layer.initialize(X, Y)
    return model


def resize(model, new_nO, resizable_layer, *, fill_defaults=None):
    old_layer = resizable_layer.layers[0]
    if old_layer.has_dim("nO") is None:
        # the output layer had not been initialized/trained yet
        old_layer.set_dim("nO", new_nO)
        return model
    elif new_nO == old_layer.get_dim("nO"):
        # the output dimension didn't change
        return model

    # initialize the new layer
    layer_creation = resizable_layer.attrs["layer_creation"]
    new_layer = layer_creation()
    new_layer.set_dim("nO", new_nO)
    if old_layer.has_dim("nI"):
        new_layer.set_dim("nI", old_layer.get_dim("nI"))
    new_layer.initialize()

    if old_layer.has_param("W"):
        larger_W = new_layer.get_param("W")
        larger_b = new_layer.get_param("b")
        smaller_W = old_layer.get_param("W")
        smaller_b = old_layer.get_param("b")
        # copy the original weights
        larger_W[: len(smaller_W)] = smaller_W
        larger_b[: len(smaller_b)] = smaller_b
        # ensure that the new weights do not influence predictions
        if fill_defaults and "W" in fill_defaults:
            larger_W[len(smaller_W) :] = fill_defaults["W"]
        if fill_defaults and "b" in fill_defaults:
            larger_b[len(smaller_b) :] = fill_defaults["b"]
        new_layer.set_param("W", larger_W)
        new_layer.set_param("b", larger_b)

    resizable_layer.layers[0] = new_layer
    return model
