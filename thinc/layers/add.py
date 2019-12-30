from typing import Tuple, Callable, List, Optional

from ..model import Model
from ..types import Array
from ..util import get_width


def add(layers: List[Model]) -> Model:
    if layers and layers[0].name == "add":
        layers[0].layers.extend(layers[1:])
        return layers[0]
    return Model(
        "add",
        forward,
        init=init,
        dims={"nO": None, "nI": None},
        params={},
        layers=[],
        attrs={},
    )


InputType = Array
OutputType = Array


def forward(model: Model, X: InputType, is_train: bool) -> Tuple[OutputType, Callable]:
    Ys, callbacks = zip(*[lyr(X, is_train=is_train) for lyr in model.layers])
    Y = Ys[0]
    for y in Ys:
        Y += y

    def finish_update_add(d_output: OutputType) -> InputType:
        grads = [bp(d_output) for bp in callbacks]
        if grads:
            total = grads[0]
            for g in grads:
                total += g
            return total
        else:
            return None

    return Y, finish_update_add


def init(model: Model, X: Optional[Array] = None, Y: Optional[Array] = None) -> None:
    if X is not None:
        X_width = get_width(X)
        model.set_dim("nI", X_width)
        for layer in model.layers:
            layer.set_dim("nI", X_width)
    for layer in model.layers:
        layer.initialize(X=X)
    model.set_dim("nO", sum(layer.get_dim("nO") for layer in model._layers))
