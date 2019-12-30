from typing import Tuple, Callable, List, Optional, TypeVar

from ..model import Model
from ..types import Array
from ..util import get_width


InputType = TypeVar("InputType", bound=Array)
OutputType = TypeVar("OutputType", bound=Array)


def concatenate(layers: List[Model]) -> Model:
    if layers and layers[0].name == "concatenate":
        layers[0].layers.extend(layers[1:])
        return layers[0]
    return Model("concatenate", forward, init=init, dims={"nO": None, "nI": None})


def forward(model: Model, X: InputType, is_train: bool) -> Tuple[OutputType, Callable]:
    Ys, callbacks = zip(*[lyr(X, is_train=is_train) for lyr in model.layers])
    widths = [Y.shape[1] for Y in Ys]
    output = model.ops.xp.hstack(Ys)

    def backprop(d_output: OutputType) -> InputType:
        dX = callbacks[0](d_output[: widths[0]])
        start = widths[0]
        for bwd, width in zip(callbacks[1:], widths[1:]):
            dX += bwd(d_output[:, start : start + width])
            start += width
        return dX

    return output, backprop


def init(
    model: Model, X: Optional[InputType] = None, Y: Optional[OutputType] = None
) -> None:
    if X is not None:
        X_width = get_width(X)
        model.set_dim("nI", X_width)
        for layer in model.layers:
            layer.set_dim("nI", X_width)
    for layer in model.layers:
        layer.initialize(X=X)
    model.set_dim("nO", sum(layer.get_dim("nO") for layer in model.layers))
