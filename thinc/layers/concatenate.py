from typing import Tuple, Callable, Optional, TypeVar, cast

from ..model import Model
from ..config import registry
from ..types import Array2d
from ..util import get_width
from .noop import noop


InT = TypeVar("InT", bound=Array2d)
OutT = TypeVar("OutT", bound=Array2d)


@registry.layers("concatenate.v0")
def concatenate(*layers: Model) -> Model[InT, OutT]:
    """Compose two or more models `f`, `g`, etc, such that their outputs are
    concatenated, i.e. `concatenate(f, g)(x)` computes `hstack(f(x), g(x))`.
    Also supports chaining more than 2 layers.
    """
    if not layers:
        return cast(Model[InT, OutT], noop())
    elif len(layers) == 1:
        return layers[0]
    elif layers[0]._func is forward:
        layers[0].layers.extend(layers[1:])
        return layers[0]

    return Model(
        "|".join(layer.name for layer in layers),
        forward,
        init=init,
        dims={"nO": None, "nI": None},
        layers=layers,
    )


def forward(model: Model[InT, OutT], X: InT, is_train: bool) -> Tuple[OutT, Callable]:
    Ys, callbacks = zip(*[lyr(X, is_train=is_train) for lyr in model.layers])
    widths = [Y.shape[1] for Y in Ys]
    output = model.ops.xp.hstack(Ys)

    def backprop(d_output: OutT) -> InT:
        dY = model.ops.as_contig(d_output[:, : widths[0]])
        dX = callbacks[0](dY)
        start = widths[0]
        for bwd, width in zip(callbacks[1:], widths[1:]):
            dY = model.ops.as_contig(d_output[:, start : start + width])
            dX += bwd(dY)
            start += width
        return dX

    return output, backprop


def init(
    model: Model[InT, OutT], X: Optional[InT] = None, Y: Optional[OutT] = None
) -> None:
    if X is not None:
        X_width = get_width(X)
        model.set_dim("nI", X_width)
        for layer in model.layers:
            layer.set_dim("nI", X_width)
    for layer in model.layers:
        layer.initialize(X=X, Y=Y)
    model.set_dim("nO", sum(layer.get_dim("nO") for layer in model.layers))
