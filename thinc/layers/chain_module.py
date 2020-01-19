from typing import Tuple, Callable, Optional, TypeVar, Any

from ..model import Model
from ..config import registry
from ..util import get_width
from ..types import Ragged, Padded, Reduced_OutT


InT = TypeVar("InT")
OutT = TypeVar("OutT")
Mid1T = TypeVar("Mid1T")
Mid2T = TypeVar("Mid2T")


# TODO: Unhack this when we can
# We currently have an issue with Pydantic when arguments have generic types.
# https://github.com/samuelcolvin/pydantic/issues/1158
# For now we work around the issue by applying the decorator to this blander
# version of the function.
@registry.layers("chain.v0")
def chain_no_types(*layer: Model) -> Model:
    return chain(*layer)


def chain(
    layer1: Model[InT, Mid1T], layer2: Model[Mid1T, OutT], *layers: Model
) -> Model[InT, Reduced_OutT]:
    """Compose two models `f` and `g` such that they become layers of a single
    feed-forward model that computes `g(f(x))`.
    Also supports chaining more than 2 layers.
    """
    layers = (layer1, layer2) + layers
    model: Model[InT, Any] = Model(
        ">>".join(layer.name for layer in layers),
        forward,
        init=init,
        dims={"nO": None, "nI": None},
        layers=layers,
    )
    return model


def forward(model: Model[InT, OutT], X: InT, is_train: bool) -> Tuple[OutT, Callable]:
    """Apply the layers of `model` in sequence, feeding the output from one
    layer into the next.
    """
    callbacks = []
    for layer in model.layers:
        Y, inc_layer_grad = layer(X, is_train=is_train)
        callbacks.append(inc_layer_grad)
        X = Y

    def backprop(dY: OutT) -> InT:
        for callback in reversed(callbacks):
            dX = callback(dY)
            dY = dX
        return dX

    return Y, backprop


def init(
    model: Model[InT, OutT], X: Optional[InT] = None, Y: Optional[OutT] = None
) -> Model[InT, OutT]:
    if X is None and Y is None:
        for layer in model.layers:
            layer.initialize()
        if model.layers[0].has_dim("nI"):
            model.set_dim("nI", model.layers[0].get_dim("nI"))
        if model.layers[-1].has_dim("nO"):
            model.set_dim("nO", model.layers[-1].get_dim("nO"))
        return model
    # Try to set nO on each layer, where available.
    # Shape inference is tricky, especially for the output. The policy is:
    # if a layer doesn't expose a nO dim, then its output is assumed to be
    # the same as its input.
    nO = None
    if Y is not None and isinstance(Y, (Ragged, Padded, model.ops.xp.ndarray, list)):
        nO = get_width(Y)
    elif model.has_dim("nO"):
        nO = model.get_dim("nO")
    # TODO: This sort of doesn't work currently -- we only get Y passed through
    # for the last layer, but maybe we need it for the second last (e.g. if we
    # have a transform at the end. Not sure what to do.
    for layer in reversed(model.layers):
        if nO is not None and layer.has_dim("nO") is None:
            layer.set_dim("nO", nO)
        if layer.has_dim("nI"):
            nO = layer.get_dim("nI")
        else:
            break
    for i, layer in enumerate(model.layers):
        if layer.has_dim("nO") is None:
            # If we're the last layer with an nO, use Y.
            if all(lyr.has_dim("nO") is False for lyr in model.layers[i + 1 :]):
                layer.initialize(X=X, Y=Y)
            else:
                layer.initialize(X=X)
        else:
            layer.initialize(X=X)
        if X is not None:
            X = layer.predict(X)
    if model.layers[0].has_dim("nI"):
        model.set_dim("nI", model.layers[0].get_dim("nI"))
    layers_with_nO = [lyr for lyr in model.layers if lyr.has_dim("nO")]
    if layers_with_nO:
        model.set_dim("nO", layers_with_nO[-1].get_dim("nO"))
    return model
